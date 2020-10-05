import torch
import torch.nn as nn
from typing import Dict, List
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.models import Model
from transformers.modeling_bert import BertLayerNorm
from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure
from .layers.gcn import GraphConvolution
import copy
from overrides import overrides


@Model.register("asbigcn")
class ASBIGCN(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 hidden_dim,
                 **kwargs):
        super().__init__(vocab, **kwargs)
        out_class = vocab.get_vocab_size('labels')
        T = 3
        self._text_embed_dropout = nn.Dropout(0.3)
        self._text_field_embedder = text_field_embedder
        self._dual_transformers = nn.ModuleList([copy.deepcopy(DualTransformer(hidden_dim)) for _ in range(T)])
        self._W_plum = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
        self._W3 = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self._final_classifier = nn.Linear(hidden_dim, out_class)
        self._loss_fn = nn.CrossEntropyLoss()
        self._positive_class_f1 = FBetaMeasure(average='macro')
        self._accuracy = CategoricalAccuracy()
    
    def forward(self,
                tokens: TextFieldTensors,
                adj_in: torch.Tensor,
                adj_out: torch.Tensor,
                transformer_indices: List[List],
                span_indices: List[List],
                label: torch.IntTensor = None
                ) -> Dict[str, torch.Tensor]:
        # embeddins
        embedded_text = self._text_field_embedder(tokens)  # b * words * 768
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
        # S_tr, S_g, adj_in, adj_out, mask
        for i, l in enumerate(self._dual_transformers):
            if i == 0:  
                S_tr, S_g = l(Ns, Ns, adj_in, adj_out, get_padding_mask(Ns_lengths, max_Ns_len))
            else:
                S_tr, S_g = l(S_tr, S_g, adj_in, adj_out, get_padding_mask(Ns_lengths, max_Ns_len))
        
        # aspect span
        max_spans = max([len(spans) for spans in span_indices])
        h_f = torch.zeros(batch_size, max_spans, S_tr.shape[2]).float().cuda()  # (b, max_spans, hidden)
        for i, spans in enumerate(span_indices):
            for j, span in enumerate(spans):
                # MaxPooling
                span_mat = S_tr[i, span[0]:span[1]]
                h_f[i, j], _ = torch.max(span_mat, dim=-2)
    
        h_f, _ = torch.max(h_f, -2)  # (b, hidden)
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
        probs = nn.functional.softmax(logits, -1)  # (b, r_class)
        output = {"logits": logits, "probs": probs}
        
        if label is not None:
            self._accuracy(logits, label)
            self._positive_class_f1(logits, label)
            output["loss"] = self._loss_fn(logits, label)
        
        return output

#     @overrides
#     def make_output_human_readable(
#             self, output_dict: Dict[str, torch.Tensor]
#     ) -> Dict[str, torch.Tensor]:
#         return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        accuracy = self._accuracy.get_metric(reset)
        f1 = self._positive_class_f1.get_metric(reset)["fscore"]

        return {
            "accuracy": accuracy,
            "f1": f1
        }
    
    
class BiGCN(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self._gcn_in = GraphConvolution(hidden_size, hidden_size, bias=False)
        self._gcn_out = GraphConvolution(hidden_size, hidden_size, bias=False)
        self.fc_O = nn.Linear(2*hidden_size, hidden_size)
        self._layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, Q_t, adj_out, adj_in):
        """
        :param Q_t: b, Ns, hidden
        :param adj_out: b, Ns, Ns
        :param adj_in: b, Ns, Ns
        :return: (b, Ns, hidden)
        """
        Q_out = self._gcn_out(Q_t, adj_out)  # b, Ns, hidden
        Q_in = self._gcn_in(Q_t, adj_in)  # b, Ns, hidden
        concat = torch.cat([Q_out, Q_in], dim=-1)  # b, Ns, 2*hidden
        concat = nn.functional.relu(self.fc_O(concat))  # b, Ns, hidden
        Q_t_next = self._layer_norm(Q_t+concat)  # b, Ns, hidden
        return Q_t_next


class DualTransformer(nn.Module):
    def __init__(self, input_dim):
        super(DualTransformer, self).__init__()
        self._transformer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=8)
        self._bi_gcn = BiGCN(hidden_size=input_dim)
        self._bi_affine = BiAffine(hidden_size=input_dim)
        
        # layer norm for graph and flat
        self._layer_norm_g = torch.nn.LayerNorm(input_dim)
        self._layer_norm_f = torch.nn.LayerNorm(input_dim)
    
    def renorm(self, adj1, adj2):
        adj = adj1 * adj2
        adj = adj / (adj.sum(-1).unsqueeze(-1) + 1e-10)
        return adj
    
    def forward(self, S_tr, S_g, adj_in, adj_out, mask):
        """
        :param S_tr: (b, max_Ns, 768)
        :param adj_in: (b, Ns, Ns)
        :param adj_out: (b, Ns, Ns)
        :param mask:  (b, max_Ns)
        :return:
        """
        S_tr = torch.transpose(S_tr, 0, 1)
        S_tr1 = self._transformer(src=S_tr, src_key_padding_mask=mask)
        S_tr1 = self._transformer(src=S_tr)
        S_tr1 = torch.transpose(S_tr1, 0, 1)
        S_g1 = self._bi_gcn(S_g, adj_out, adj_in)
        S_tr2, S_q2 = self._bi_affine(S_tr1, S_g1)
        S_tr_out = self._layer_norm_f(S_tr1 + S_tr2)
        S_g_out = self._layer_norm_g(S_g1 + S_q2)
        return S_tr_out, S_g_out


class BiAffine(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, S1, S2):
        """
        :param S1: S_tr (b, N1, hidden)
        :param S2: S_g (b, N2, hidden)
        :return:
        """
        S2_h = self.W1(S2)  # (b, N2, hidden)
        matmul1 = torch.matmul(S1, torch.transpose(S2_h, 1, 2))
        attn1 = torch.softmax(matmul1, dim=-1)  # (b, N1, N2)
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

def get_padding_mask(lengths, max_length):
    """
    :param lengths: b
    :param max_length
    :return: b * max_length
    """
    lengths = lengths.unsqueeze(-1).repeat([1] * lengths.dim() + [max_length]).long()
    range = torch.arange(max_length).cuda()
    range = range.expand_as(lengths)
    mask = range > lengths
    return mask

