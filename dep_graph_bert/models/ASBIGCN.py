from typing import List
import torch
import torch.nn as nn
from typing import Dict
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.models import Model
from transformers.modeling_bert import BertLayerNorm
from dep_graph_bert.models.layers.dynamic_rnn import DynamicLSTM
from .layers.gcn import GraphConvolution


@Model.register("asbigcn")
class ASBIGCN(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 embedding_matrix,
                 opt,
                 **kwargs):
        super().__init__(vocab, **kwargs)
        out_class = vocab.get_vocab_size('label')
        
        self._text_embed_dropout = nn.Dropout(0.1)
        self._text_field_embedder = text_field_embedder
        self.opt = opt
        self.text_lstm = DynamicLSTM(input_size=768, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)
        self._dual_transformer = DualTransformer(opt.hidden_dim, opt.hidden_dim, opt.edge_size, bias=True)
        self._W_plum = nn.Linear(2 * opt.hidden_dim, opt.hidden_dim, bias=False)
        self._W3 = torch.nn.Linear(opt.hidden_dim, opt.hidden_dim, bias=False)
        self._final_classifier = nn.Linear(opt.hidden_dim, out_class)
    
    def forward(self,
                tokens: TextFieldTensors,
                transformer_indices: List,
                arg_matrix: torch.Tensor,
                aspect_span: torch.Tensor,
                label: torch.IntTensor = None
                ) -> Dict[str, torch.Tensor]:
        """
        :param tokens:
        :param transformer_indices: b - Ns_len - 2(start, end)
        :param arg_matrix:
        :param aspect_span:
        :param label:
        :return:
        """

        text_indices, span_indices, transformer_indices, adj1, adj2, edge1, edge2 = inputs
        ################
        # text_indices = b * len (token_ids)
        # span_indices = b * num_aspects
        # transformer_indices = b - Ns_len - 2(start, end)
        ################
        # embeddins
        embedded_text = self._text_field_embedder(tokens)  # b * words * 768
        embedded_text = embedded_text[:, 1:, :]  # remove [CLS]
        embedded_text = self._text_embed_dropout(embedded_text)
        batch_size = embedded_text.shape[0]
        
        max_Ns_len = max([len(spans) for spans in transformer_indices])
        Ns = torch.zeros(batch_size, max_Ns_len, 768).float().cuda() # b * max_Ns * 768
        
        # sum aspect spans
        for i, spans in enumerate(transformer_indices):
            for j, span in enumerate(spans):
                Ns[i, j] = torch.sum(embedded_text[i, span[0]:span[1]], 0)
        Ns = self._text_embed_dropout(Ns)
        
        # Dual-transformer structure
        Ns_lengths = torch.Tensor([len(item) for item in transformer_indices]).long().cuda()
        S_tr, _ = self._dual_transformer(Ns, adj1, adj2, edge1, edge2, length2mask(Ns_lengths, max_Ns_len))  # b, Ns, hidden
        
        # aspect span
        max_spans = max([len(spans) for spans in span_indices])
        h_f = torch.zeros(batch_size, max_spans, self.opt.hidden_dim).float().cuda()  # (b, max_spans, hidden)
        for i, spans in enumerate(span_indices):
            for j, span in enumerate(spans):
                # MaxPooling
                h_f[i, j], _ = torch.max(S_tr[i, span[0]:span[1]], -2)
    
        h_f = h_f[:, 0, :]  # (b, hidden)
        h_f = self._W3(h_f)  # (b, hidden)
        
        # mask out aspects' tokens
        masked = length2mask(Ns_lengths, max_Ns_len)  # b * max_Ns_len
        for i, spans in enumerate(span_indices):
            for j, span in enumerate(spans):
                masked[i, span[0]:span[1]] = 0
        
        masked = (1 - masked) * -1e20
        masked = masked.unsqueeze(-2)  # b, 1, Ns
        attn_weights = torch.matmul(h_f.unsqueeze(1), S_tr.transpose(1, 2))  # b, 1, Ns
        attn_weights = nn.functional.softmax(attn_weights + masked, dim=-1)  # b, 1, Ns
        S_tr_attn = torch.matmul(attn_weights, S_tr).squeeze(1)  # b * hidden
        
        # final classifier
        h_f_last = nn.functional.relu(self._W_plum(torch.cat([h_f, S_tr_attn], -1)))  # b, hidden
        logits = self._final_classifier(h_f_last)  # (b, r_class)
        probs = nn.functional.softmax(logits)  # (b, r_class)
        output = {"logits": logits, "probs": probs}
        
        if label is not None:
            output["loss"] = self._loss_fn(logits, label)
        
        return output
    
    
class BiGCN(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self._gcn_out = GraphConvolution(hidden_size, hidden_size, bias=False)
        self._gcn_in = GraphConvolution(hidden_size, hidden_size, bias=False)
        self.fc_O = nn.Linear(2*hidden_size, hidden_size)
        self._layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, Q_t, adj_in, adj_out):
        """
        :param Q_t: b, Ns, hidden
        :param adj_in: b, Ns, Ns
        :param adj_out: b, Ns, Ns
        :return: (b, Ns, hidden)
        """
        Q_out = self._gcn_out(Q_t, adj_out)  # b, Ns, hidden
        Q_in = self._gcn_in(Q_t, adj_in)  # b, Ns, hidden
        concat = torch.cat([Q_out, Q_in], dim=-1)  # b, Ns, 2*hidden
        concat = nn.functional.relu(self.fc_O(concat))  # b, Ns, hidden
        Q_t_next = self._layer_norm(Q_t+concat)  # b, Ns, hidden
        return Q_t_next


class SelfAlignment(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    
    def __init__(self, in_features, bias=True):
        super(SelfAlignment, self).__init__()
        self.in_features = in_features
        self.dropout = nn.Dropout(0.1)
        #        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, in_features))
        self.linear = torch.nn.Linear(in_features, in_features, bias=False)
        self.linear1 = torch.nn.Linear(in_features, in_features)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(in_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, text, text1, textmask):
        logits = torch.matmul(self.linear(text), text1.transpose(1, 2))
        masked = textmask.unsqueeze(1)
        masked = (1 - masked) * -1e20
        logits = torch.softmax(logits + masked, -1)
        output = torch.matmul(logits, text1)
        #        output = self.dropout(torch.relu(self.linear1(torch.matmul(logits,text1))))+text
        output = output * textmask.unsqueeze(-1)
        if self.bias is not None:
            return output + self.bias, logits * textmask.unsqueeze(-1)
        else:
            return output, logits * textmask.unsqueeze(-1)


def init_weights(module):
    """ Initialize the weights.
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=0.01)
    elif isinstance(module, BertLayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class DualTransformer(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    
    def __init__(self, input_dim, output_dim, edge_size, bias=False):
        super(DualTransformer, self).__init__()
        self.T = 3
        #        self.bertlayer=BertLayer(config)
        #        self.bertlayer1=BertcoLayer(config)
        #        self.bertlayer.apply(init_weights)
        self._layer_norm = torch.nn.LayerNorm(output_dim)
        self.edge_vocab = torch.nn.Embedding(edge_size, output_dim)
        self._input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = nn.Dropout(0.1)
        #        self.dropout1 = nn.Dropout(0.0)
        self._transform_input2output_dim = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self._W = torch.nn.Linear(output_dim, output_dim, bias=False)
        #        self.linear2=torch.nn.Linear(4*out_features,out_features,bias=False)
        self.align = SelfAlignment(output_dim)
        #        self.align1=selfalignment(out_features)
        #        self.align1=selfalignment(out_features)
        #        self.align2=selfalignment(out_features)
        self._bi_gcn = BiGCN(hidden_size=hidden_dim)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('bias', None)
    
    def renorm(self, adj1, adj2):
        adj = adj1 * adj2
        adj = adj / (adj.sum(-1).unsqueeze(-1) + 1e-10)
        return adj
    
    def forward(self, embedded_text, adj_in, adj_out, textmask):
        """
        :param embedded_text: (b, max_Ns, 768)
        :param adj_in:
        :param adj_out:
        :param edge1:
        :param edge2:
        :param textmask:  (b, max_Ns)
        :return:
        """
        adj = adj_in + adj_out
        adj[adj >= 1] = 1
        attss = [adj_in, adj_out]
        
        # Transform dimension if input and output dimensions are not equal
        if self._input_dim != self.output_dim:
            output = torch.relu(torch.matmul(embedded_text, self._transform_input2output_dim))
            out = torch.relu(torch.matmul(embedded_text, self._transform_input2output_dim))
            outss = out
        else:
            output = embedded_text
            out = embedded_text
            outss = embedded_text
        #        outs,att1=self.align(out,out, textmask)
        #        attss.append(att1)
        
        # Perform convolution
        for _ in range(self.T):
            #            outs=self.bertlayer(out,attention_mask=extended_attention_mask)[0]
            #            outs,att1=self.align(out,out, textmask)
            #            attss.append(att1)
            outs = output
            #            text2=output.unsqueeze(-3).repeat(1,textlen,1,1)
            #            teout=torch.matmul(text2.unsqueeze(-2),edge).squeeze(-2)
            #            teout=self.linear2(torch.cat([edge,text2,edge*text2,torch.abs(edge-text2)],-1))
            
            teout = self._W(output)
            denom1 = torch.sum(adj, dim=2, keepdim=True) + 1
            ##            atts=self.renorm(adj,att1)
            output = self.dropout(torch.relu(torch.matmul(adj, teout) / denom1))
            #            output = self.dropout(torch.relu((adj.unsqueeze(-1)*teout).sum(-2) / denom1))
            #            denom2 = torch.sum(adj2, dim=2, keepdim=True)+1
            ##            atts=self.renorm(adj,att1)
            #            output2 = self.dropout(torch.relu(torch.matmul(adj2,teout) / denom2))
            #            output=self.dropout1(torch.relu(self.linear2(torch.cat([output1,output2],-1))))
            if self.bias is not None:
                output = output + self.bias
            output = self._layer_norm(output) + outs
            outs = output
        #            out=outs+output
        #            output=out
        #            outss=outs+output+self.align(output,outs, textmask)[0]*0
        #            out=self.bertlayer1(outs,output, attention_mask=extended_attention_mask)[0]
        #            outss=self.bertlayer1(output,outs, attention_mask=extended_attention_mask)[0]
        #            out,_=self.align(out1,output, textmask)
        #        out,att1=self.align1(output+outs,output+outs, textmask)
        #        attss.append(att1)
        outs1 = outs
        outs, att1 = self.align(outs, outs, textmask)
        outs = torch.cat([outs, outs1], -1)
        return outs, attss  # b,s,h


class biedgeGraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    
    def __init__(self, in_features, out_features, edge_size, bias=True):
        super(biedgeGraphConvolution, self).__init__()
        self.K = 3
        self.norm = torch.nn.LayerNorm(out_features)
        self.edge_vocab = torch.nn.Embedding(edge_size, out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = nn.Dropout(0.3)
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.fuse1 = nn.Sequential(torch.nn.Linear(2 * out_features, out_features, bias=False), torch.nn.ReLU())
        self.fuse2 = nn.Sequential(torch.nn.Linear(2 * out_features, out_features, bias=False), torch.nn.ReLU())
        self.fc3 = nn.Sequential(torch.nn.Linear(2 * out_features, out_features, bias=False), torch.nn.ReLU())
        self.fc1 = nn.Sequential(torch.nn.Linear(3 * out_features, out_features, bias=False), torch.nn.ReLU(),
                                 torch.nn.Linear(out_features, 1, bias=True))
        self.fc2 = nn.Sequential(torch.nn.Linear(3 * out_features, out_features, bias=False), torch.nn.ReLU(),
                                 torch.nn.Linear(out_features, 1, bias=True))
        self.fc1s = nn.Sequential(torch.nn.Linear(3 * out_features, out_features, bias=False), torch.nn.ReLU(),
                                  torch.nn.Linear(out_features, 1, bias=True))
        self.fc2s = nn.Sequential(torch.nn.Linear(3 * out_features, out_features, bias=False), torch.nn.ReLU(),
                                  torch.nn.Linear(out_features, 1, bias=True))
        self.align = SelfAlignment(out_features)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
    
    def renorm(self, adj1, adj2, a=0.5):
        adj = adj1 + a * adj2
        adj = adj / (adj.sum(-1).unsqueeze(-1) + 1e-20)
        return adj
    
    def forward(self, text, adj1, adj2, edge1, edge2, textmask):
        #        print(edge1.size())
        #        edge1=self.edge_vocab(edge1)#b,s,s,h
        #        edge2=self.edge_vocab(edge2)#b,s,s,h
        textlen = text.size(1)  # s
        outss, att1 = self.align(text, text, textmask)
        output = torch.relu(torch.matmul(text, self.weight))
        #        adj1s=copy.deepcopy(adj1)
        #        adj2s=copy.deepcopy(adj2)
        #        for i in range(adj1s.size(1)):
        #            adj1s[:,i,i]=0
        #            adj2s[:,i,i]=0
        for i in range(self.K):
            text1 = output.unsqueeze(-2).repeat(1, 1, textlen, 1)
            text2 = output.unsqueeze(-3).repeat(1, textlen, 1, 1)
            teout = self.fuse1(torch.cat([text2, edge1], -1))  # b,s,s,h
            tein = self.fuse2(torch.cat([text2, edge2], -1))  # b,s,s,h
            teouts = torch.sigmoid(self.fc1(torch.cat([text1, text2, edge1], -1)))  # b,s,s,1
            teins = torch.sigmoid(self.fc2(torch.cat([text1, text2, edge2], -1)))  # b,s,s,1
            #            teoutss=torch.softmax((1-adj1s)*-1e20+self.fc1s(torch.cat([text1,text2,edge1],-1)).squeeze(-1),-1)#b,s,s,1
            #            teinss=torch.softmax((1-adj2s)*-1e20+self.fc2s(torch.cat([text1,text2,edge2],-1)).squeeze(-1),-1)#b,s,s,1
            #            for i in range(adj1s.size(1)):
            #                teoutss.data[:,i,i]=1.0
            #                teinss.data[:,i,i]=1.0
            #            hidden1 = torch.matmul(text, self.weight)
            denom1 = torch.sum(adj1, dim=2, keepdim=True) + 1
            #            adj1s=self.renorm(att1,adj1)
            #            output1 = torch.sum(adj1.unsqueeze(-1)*teout*teouts*teoutss.unsqueeze(-1),-2) / denom1
            output1 = torch.sum(adj1.unsqueeze(-1) * teout * teouts, -2) / denom1
            denom2 = torch.sum(adj2, dim=2, keepdim=True) + 1
            #            adj2s=self.renorm(att1,adj2)
            #            output2 = torch.sum(adj2.unsqueeze(-1)*tein*teins*teinss.unsqueeze(-1),-2) / denom2
            output2 = torch.sum(adj2.unsqueeze(-1) * tein * teins, -2) / denom2
            output = self.fc3(torch.cat([output1, output2], -1)) + output
            if self.bias is not None:
                output = output + self.bias
            output = self.dropout(self.norm(output))
        return output, outss  # b,s,h


class BiAffine(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.W1 = nn.Linear(hidden, hidden, bias=False)
        self.W2 = nn.Linear(hidden, hidden, bias=False)

    def forward(self, S1, S2):
        """
        :param S1: S_tr (b, N1, hidden)
        :param S2: S_g (b, N2, hidden)
        :return:
        """
        S2_h = self.W1(S2)  # (b, N2, hidden)
        attn1 = torch.softmax(torch.matmul(S1, S2_h.transpose(1, 2)), dim=-1)  # (b, N1, N2)
        S1_out = torch.matmul(attn1, S2)  # (b, N1, hidden)
        
        S1_h = self.W2(S1)  # (b, N1, hidden)
        attn2 = torch.softmax(torch.matmul(S2, S1_h.transpose(1, 2)), dim=-1)  # (b, N2, N1)
        S2_out = torch.matmul(attn2, S1)  # (b, N2, hidden)
        
        return S1_out, S2_out


def length2mask(lengths, max_length):
    """
    :param lengths: b
    :param max_length
    :return: b * max_length
    """
    lengths = lengths.unsqueeze(-1).repeat([1] * lengths.dim() + [max_length]).long()
    range = torch.arange(max_length).cuda()
    range = range.expand_as(lengths)
    mask = range < lengths
    return mask.float().cuda()





