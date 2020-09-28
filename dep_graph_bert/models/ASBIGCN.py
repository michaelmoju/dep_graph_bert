import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, DefaultDict, Set, Union
import allennlp.nn.util as util
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.models import Model
from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure
from overrides import overrides
from transformers.modeling_bert import BertLayerNorm, BertModel
from .dynamic_rnn import DynamicLSTM


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


class simpleGraphConvolutionalignment(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    
    def __init__(self, in_features, out_features, edge_size, bias=False):
        super(simpleGraphConvolutionalignment, self).__init__()
        self.K = 3
        #        self.bertlayer=BertLayer(config)
        #        self.bertlayer1=BertcoLayer(config)
        #        self.bertlayer.apply(init_weights)
        self.norm = torch.nn.LayerNorm(out_features)
        self.edge_vocab = torch.nn.Embedding(edge_size, out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = nn.Dropout(0.1)
        #        self.dropout1 = nn.Dropout(0.0)
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.linear = torch.nn.Linear(out_features, out_features, bias=False)
        #        self.linear2=torch.nn.Linear(4*out_features,out_features,bias=False)
        self.align = SelfAlignment(out_features)
        #        self.align1=selfalignment(out_features)
        #        self.align1=selfalignment(out_features)
        #        self.align2=selfalignment(out_features)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
    
    def renorm(self, adj1, adj2):
        adj = adj1 * adj2
        adj = adj / (adj.sum(-1).unsqueeze(-1) + 1e-10)
        return adj
    
    def forward(self, embedded_text, adj1, adj2, edge1, edge2, textmask):
        """
        :param embedded_text: S0, contextualized embedding, b * sent_len * 768
        :param adj1:
        :param adj2:
        :param edge1:
        :param edge2:
        :param textmask:
        :return:
        """
        adj = adj1 + adj2
        adj[adj >= 1] = 1
        edge = self.edge_vocab(edge1 + edge2)
        #        edge=edge.view(-1,edge.size(1),edge.size(1),self.out_features,self.out_features)
        #        for i in range(adj.size(1)):
        #            adj[:,i,i]=0
        #        adj=adj2
        textlen = embedded_text.size(1)  # s
        #        extended_attention_mask = textmask.unsqueeze(1).unsqueeze(2)
        #        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        attss = [adj1, adj2]
        if (self.in_features != self.out_features):
            output = torch.relu(torch.matmul(embedded_text, self.weight))
            out = torch.relu(torch.matmul(embedded_text, self.weight))
            outss = out
        else:
            output = embedded_text
            out = embedded_text
            outss = embedded_text
        #        outs,att1=self.align(out,out, textmask)
        #        attss.append(att1)
        for _ in range(self.K):
            #            outs=self.bertlayer(out,attention_mask=extended_attention_mask)[0]
            #            outs,att1=self.align(out,out, textmask)
            #            attss.append(att1)
            outs = output
            #            text2=output.unsqueeze(-3).repeat(1,textlen,1,1)
            #            teout=torch.matmul(text2.unsqueeze(-2),edge).squeeze(-2)
            #            teout=self.linear2(torch.cat([edge,text2,edge*text2,torch.abs(edge-text2)],-1))
            
            teout = self.linear(output)
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
            output = self.norm(output) + outs
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


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output


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


class ASGCN(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(ASGCN, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.text_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.gc1 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.gc2 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.fc = nn.Linear(2 * opt.hidden_dim, opt.polarities_dim)
        self.text_embed_dropout = nn.Dropout(0.3)
    
    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i, 0]):
                weight[i].append(1 - (aspect_double_idx[i, 0] - j) / context_len)
            for j in range(aspect_double_idx[i, 0], aspect_double_idx[i, 1] + 1):
                weight[i].append(0)
            for j in range(aspect_double_idx[i, 1] + 1, text_len[i]):
                weight[i].append(1 - (j - aspect_double_idx[i, 1]) / context_len)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        weight = torch.tensor(weight).unsqueeze(2).to(self.opt.device)
        return weight * x
    
    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i, 0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i, 0], aspect_double_idx[i, 1] + 1):
                mask[i].append(1)
            for j in range(aspect_double_idx[i, 1] + 1, seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask).unsqueeze(2).float().to(self.opt.device)
        return mask * x
    
    def forward(self, inputs):
        text_indices, aspect_indices, left_indices, adj = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len + aspect_len - 1).unsqueeze(1)], dim=1)
        text = self.embed(text_indices)
        text = self.text_embed_dropout(text)
        text_out, (_, _) = self.text_lstm(text, text_len)
        x = F.relu(self.gc1(self.position_weight(text_out, aspect_double_idx, text_len, aspect_len), adj))
        x = F.relu(self.gc2(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        x = self.mask(x, aspect_double_idx)
        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, text_out).squeeze(1)  # batch_size x 2*hidden_dim
        output = self.fc(x)
        return output


class mutualatts(nn.Module):
    def __init__(self, hidden_size):
        super(mutualatts, self).__init__()
        self.linear2 = torch.nn.Linear(3 * hidden_size, hidden_size)
        self.linear1 = torch.nn.Linear(hidden_size, 1)
    
    def forward(self, in1, in2, text, textmask):
        length = text.size(1)
        in1 = in1.unsqueeze(1).repeat(1, length, 1)
        in2 = in2.unsqueeze(1).repeat(1, length, 1)
        att = self.linear1(torch.tanh(self.linear2(torch.cat([in1, in2, text], -1)))).squeeze(-1)
        att = torch.softmax(((1 - textmask) * -1e20 + att), -1).unsqueeze(1)
        context = torch.matmul(att, text).squeeze(1)
        return context, att


class mutualatt(nn.Module):
    """
    Aspect-based Attention Module
    """
    def __init__(self, hidden_size):
        super(mutualatt, self).__init__()
        self.linear2 = torch.nn.Linear(2 * hidden_size, hidden_size)
        self.linear1 = torch.nn.Linear(hidden_size, 1)
    
    def forward(self, in1, text, mask):
        length = text.size(1)
        in1 = in1.unsqueeze(1).repeat(1, length, 1)
        att = self.linear1(torch.tanh(self.linear2(torch.cat([in1, text], -1)))).squeeze(-1)
        att = torch.softmax(att, -1) * mask
        att = (att / (att.sum(-1, keepdim=True) + 1e-20)).unsqueeze(-2)
        #        att=torch.softmax(((1-textmask)*-1e20+att),-1).unsqueeze(1)
        context = torch.matmul(att, text).squeeze(1)
        return context, att


def length2mask(lengths, max_length):
    """
    :param lengths: b
    :param max_length:
    :return: b * max_length
    """
    lengths = lengths.unsqueeze(-1).repeat([1] * lengths.dim() + [max_length]).long()
    range = torch.arange(max_length).cuda()
    range = range.expand_as(lengths)
    mask = range < lengths
    return mask.float().cuda()


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
        #        self.mul1=mutualatt(2*opt.hidden_dim)
        #        self.mul2=mutualatt(2*opt.hidden_dim)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.text_lstm = DynamicLSTM(768, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        #        self.text_lstm1 = DynamicLSTM(2*opt.hidden_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        #        self.text_lstm2 = DynamicLSTM(opt.embed_dim, 384, num_layers=1, batch_first=True, bidirectional=True)
        #        self.gc1 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        #        self.gc2 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc = simpleGraphConvolutionalignment(2 * opt.hidden_dim, 2 * opt.hidden_dim, opt.edge_size, bias=True)
        self.fc_final = nn.Linear(8 * opt.hidden_dim, out_class)
        #        self.fc1 = nn.Linear(768*2,768)
        
    
    #        self.linear1=torch.nn.Linear(2*opt.hidden_dim, 2*opt.hidden_dim,bias=False)
    #        self.linear2=torch.nn.Linear(2*opt.hidden_dim, 2*opt.hidden_dim,bias=False)
    #        self.linear3=torch.nn.Linear(768, 2*opt.hidden_dim,bias=True)
    
    def forward(self,
                tokens: TextFieldTensors,
                arg_matrix: torch.Tensor,
                aspect_span: torch.Tensor,
                label: torch.IntTensor = None
                ) -> Dict[str, torch.Tensor]:

        inputs, mixed=True
        
        # embeddins
        embedded_text = self._text_field_embedder(tokens)  # b * words * 768
        embedded_text = embedded_text[:, 1:, :]  # remove [CLS]
        oss = self._text_embed_dropout(embedded_text)
        
        ####
        #### span_indices = b * num_aspects
        ####
        
        text_indices, span_indices, transformer_indices, adj1, adj2, edge1, edge2 = inputs
        
        batchhalf = text_indices.size(0) // 2
        text_len = torch.sum(text_indices != 0, dim=-1)

        max_len = max([len(item) for item in transformer_indices])
        text_len = torch.Tensor([len(item) for item in transformer_indices]).long().cuda()
        tmps = torch.zeros(text_indices.size(0), max_len, 768).float().cuda()
        for i, spans in enumerate(transformer_indices):
            for j, span in enumerate(spans):
                tmps[i, j] = torch.sum(output[i, span[0]:span[1]], 0)
        #        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        #        left_len = torch.sum(left_indices != 0, dim=-1)
        #        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len+aspect_len-1).unsqueeze(1)], dim=1)
        #        text = self.embed(text_indices)
        text = self._text_embed_dropout(tmps)
        
        text, (hout, _) = self.text_lstm(text, text_len)
        x = text
        #        text_out=torch.relu(self.linear3(text))
        hout = torch.transpose(hout, 0, 1)
        hout = hout.reshape(hout.size(0), -1)
        #        print(text_out.size())
        # graph output  outs, attss
        text, attss = self.gc(text, adj1, adj2, edge1, edge2, length2mask(text_len, max_len))
        max_num_spans = max([len(item) for item in span_indices])
        
        # aspect span
        h_f_maxPool = torch.zeros(text_indices.size(0), max_num_spans, 4 * self.opt.hidden_dim).float().cuda()
        h_f_sum = torch.zeros(text_indices.size(0), max_num_spans, 2 * self.opt.hidden_dim).float().cuda()

        for i, spans in enumerate(span_indices):
            for j, span in enumerate(spans):

                # MaxPooling
                h_f_maxPool[i, j], _ = torch.max(text[i, span[0]:span[1]], -2)

                # Sum
                h_f_sum[i, j] = torch.sum(x[i, span[0]:span[1]], -2)

                maskas = length2mask(torch.tensor([len(item) for item in span_indices]).long().cuda(), max_num_spans) # b,span

                a_f = torch.softmax(torch.matmul(h_f_maxPool, hout.unsqueeze(-1)).squeeze(-1) + (1-maskas)*-1e20, -1).unsqueeze(-2)
                x = torch.matmul(a_f, h_f_maxPool).squeeze(-2) # b,span

                x=self.linear1(x)
                x1=h_f_sum[:,0,:]
                x1=torch.matmul(torch.softmax(torch.matmul(h_f_sum[:,:,:],hout.unsqueeze(-1)).squeeze(-1)+(1-maskas)*-1e20,-1).unsqueeze(-2),h_f_sum[:,:,:]).squeeze(-2)
                x1=self.linear2(x1)
                _, (x, _) = self.text_lstm1(h_f_maxPool, torch.Tensor([len(item) for item in span_indices]).long().cuda())#b,h
                x = F.relu(self.gc1(self.position_weight(text_out, aspect_double_idx, text_len, aspect_len), adj))
                x = F.relu(self.gc2(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
                x = self.mask(x, aspect_double_idx)
                x=torch.transpose(x, 0, 1)
                x=x.reshape(x.size(0),-1)

        # mask out aspects' tokens
        masked=length2mask(text_len,max_len)

        for i,spans in enumerate(span_indices):
            for j,span in enumerate(spans):
                masked[i,span[0]:span[1]]=0


        masked = (1-masked)*-1e20
        masked = masked.unsqueeze(-2)
        alpha_mat = torch.matmul(x.unsqueeze(1), text_out.transpose(1, 2))
        self.alpha = nn.functional.softmax(masked+alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(self.alpha, text_out).squeeze(1) # b * 2 * 768
#        x,self.alpha =self.mul1(x,text_out,masked)
#        x1,self.alpha1 =self.mul2(hout,text_out,length2mask(text_len,max_len))
#        x = torch.matmul(self.alpha, text_out).squeeze(1) # batch_size x 2*hidden_dim
        alpha_mat = torch.matmul(x1.unsqueeze(1), text_out.transpose(1, 2))
        a_f = nn.functional.softmax(masked + alpha_mat.sum(1, keepdim=True), dim=2)
        x1 = torch.matmul(a_f, text_out).squeeze(1) # batch_size * 2*hidden_dim

        h_f_last = self.fc1(torch.nn.functional.relu(self.fc_final(torch.cat([h_f_maxPool[:, 0, :], h_f_sum[:, 0, :]], -1))))
        logits = self.fc_final(h_f_last)
        probs = nn.functional.softmax(logits)  # (b, num_aspects, r_class)
        output = {"logits": logits, "probs": probs}
        
        if label is not None:
            output["loss"] = self._loss_fn(logits, label)
        
        return output


