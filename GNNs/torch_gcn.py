"""GCN using DGL nn package
References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import torch
import torch.nn as nn
# from dgl.nn import GraphConv
from .graphconv_edge_weight import GraphConvEdgeWeight as GraphConv

class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden, concat='False', bias='False',
                 normalization='none'):
        super(GCN, self).__init__()
        self.layer = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, norm=normalization))
        # hidden layers
        self.layers.append(GraphConv(n_hidden, n_hidden, norm=normalization))
        # output layer
        # self.dropout = nn.Dropout(p=dropout)            
    
    def forward(self, features, g, edge_weight):
        h = features    
        for i, layer in enumerate(self.layers):
        #     # if i != 0:
        #     #     h = self.dropout(h)
            h = layer(g, h, edge_weights=edge_weight)
        return h