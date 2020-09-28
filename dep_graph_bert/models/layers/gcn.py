# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))  # W_out or W_in
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.bias = None
    
    def forward(self, text, adj):
        """
        :param text: Q_t (b, Ns, hidden)
        :param adj: (b, Ns, Ns)
        :return: (b, Ns, hidden)
        """
        hidden = torch.matmul(text, self.weight)  # b, Ns, hidden
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom  # b, Ns, hidden
        if self.bias is not None:
            return output + self.bias
        else:
            return output