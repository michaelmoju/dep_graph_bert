# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self._weight = nn.Linear(in_features, out_features, bias=False)  # W_out or W_in
    
    def forward(self, text, adj):
        """
        :param text: Q_t (b, Ns, hidden)
        :param adj: (b, Ns, Ns)
        :return: (b, Ns, hidden)
        """
        hidden = self._weight(text)  # b, Ns, hidden
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom  # b, Ns, hidden
        return output