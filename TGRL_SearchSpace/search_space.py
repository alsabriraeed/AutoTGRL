import sys
# sys.path.append(r'C:/Users/Raeed/codes/20ng/micro_graphnas')           
import torch
import torch.nn.functional as F             
from GNNs.Edge_GATConv import Edge_GATConv             
from GNNs.Edge_GCNConv import Edge_GCNConv        
from GNNs.Graph_Conv import Graph_Conv
from GNNs.CG_Conv import CG_Conv
from GNNs.Transformer_Conv import Transformer_Conv
from GNNs.Hypergraph_Conv import Hypergraph_Conv          
from GNNs.GATConv import GATConv1         
from torch.nn import Module
from torch_geometric.nn.conv import *
      

gnn_list = ["edggat", "edggcn", "cheb",  "Graph_Conv","arma", "Transformer_Conv"]   # "cheb", , this is the used one                           
act_list = ["sigmoid", "tanh", "relu"]                                  

def act_map(act):
    if act == "linear":
        return lambda x: x
    elif act == "elu":
        return F.elu
    elif act == "sigmoid":
        return torch.sigmoid
    elif act == "tanh":
        return torch.tanh
    elif act == "relu":
        return torch.nn.functional.relu
    elif act == "relu6":
        return torch.nn.functional.relu6
    elif act == "softplus":
        return torch.nn.functional.softplus
    elif act == "leaky_relu":
        return torch.nn.functional.leaky_relu
    elif act == "softmax":       
        return torch.nn.functional.softmax     
    else:
        raise Exception("wrong activate function")


def gnn_map(gnn_name, in_dim, out_dim, aggr='add' , concat=False, bias=True,edge_dim=1) -> Module:
    '''     
    :param gnn_name:     
    :param in_dim:      
    :param out_dim:   
    :param concat: for gat, concat multi-head output or not
    :return: GNN model
    '''        
    if gnn_name == "gat_4":         
        return GATConv(in_dim, out_dim, 4, concat=concat, bias=bias)
    elif gnn_name == "Graph_Conv":               
        return Graph_Conv(in_dim, out_dim, aggr='add', edge_dim=edge_dim, concat=concat, bias=bias)              
    elif gnn_name == "Transformer_Conv":                
        return Transformer_Conv(in_dim, out_dim, aggr='add',  edge_dim=edge_dim, concat=concat, bias=bias)                
    elif gnn_name == "CG_Conv":                     
        return CG_Conv(in_dim, out_dim, edge_dim=edge_dim, concat=concat, bias=bias)                     
    elif gnn_name == "Hypergraph_Conv":                     
        return Hypergraph_Conv(in_dim, out_dim, aggr='add', edge_dim=edge_dim, concat=concat, bias=bias)
    elif gnn_name == "gat_6":     
        return GATConv(in_dim, out_dim, 6, concat=concat, bias=bias)
    elif gnn_name == "edggat":    
        return Edge_GATConv(in_dim, out_dim,aggr, 1, edge_dim=edge_dim, concat=concat, bias=bias)
    elif gnn_name == "edggat_2":                   
        return Edge_GATConv(in_dim, out_dim, 2, edge_dim=edge_dim, concat=concat, bias=bias)
    elif gnn_name == "edggat_4":                 
        return Edge_GATConv(in_dim, out_dim, 4, edge_dim=edge_dim, concat=concat, bias=bias)
    elif gnn_name == "edggat_6":                 
        return Edge_GATConv(in_dim, out_dim, 6, edge_dim=edge_dim, concat=concat, bias=bias)
    elif gnn_name == "edggcn":     
        return Edge_GCNConv(in_dim, out_dim,aggr, edge_dim=edge_dim, concat=concat, bias=bias)
    elif gnn_name == "gat_8":       
        return GATConv(in_dim, out_dim, 8, concat=concat, bias=bias)    
    elif gnn_name == "gat_2":                
        return GATConv(in_dim, out_dim, 2, concat=concat, bias=bias)
    elif gnn_name in ["gat_1", "gat"]:
        return GATConv(in_dim, out_dim, 1, concat=concat, bias=bias)
    elif gnn_name == "gcn":           
        return GCNConv(in_dim, out_dim)                              
    elif gnn_name == "cheb":
        return ChebConv(in_dim, out_dim, K=2, bias=bias)
    elif gnn_name == "sage":
        return SAGEConv(in_dim, out_dim, bias=bias)
    elif gnn_name == "gated":
        return GatedGraphConv(in_dim, out_dim, bias=bias)
    elif gnn_name == "arma":
        return ARMAConv(in_dim, out_dim, bias=bias)
    elif gnn_name == "sg":
        return SGConv(in_dim, out_dim, bias=bias)
    elif gnn_name == "linear":
        return LinearConv(in_dim, out_dim, bias=bias)
    elif gnn_name == "zero":                                           
        return ZeroConv(in_dim, out_dim, bias=bias)

class LinearConv(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True):
        super(LinearConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = torch.nn.Linear(in_channels, out_channels, bias)

    def forward(self, x, edge_index, edge_weight=None):
        return self.linear(x)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class ZeroConv(Module):  
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True):
        super(ZeroConv, self).__init__()
        self.out_dim = out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels    


    def forward(self, x, edge_index, edge_weight=None):
        return torch.zeros([x.size(0), self.out_dim]).to(x.device)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class IncrementSearchSpace(object):
    def __init__(self, search_space=None, max_cell=10):
        if search_space:
            self.search_space = search_space
        else:
            self.search_space = {}
            self.search_space["act"] = act_list   
            self.search_space["gnn"] = gnn_list  
            for i in range(max_cell):
                self.search_space[f"self_index_{i}"] = list(range(2+i))                   
            self.search_space["concat_type"] = ["add", "product", "concat"]                   
            self.search_space['learning_rate'] = [1e-3, 1e-2, 5e-3,1e-4,5e-4]                               
            self.search_space['dropout'] = [ 0.5, 0.6, 0.7]           
            self.search_space['weight_decay'] = [5e-5]                                                                               
            self.search_space['hidden_unit'] = [16, 32,64,128]                   
            # self.search_space['balance_factor'] = [0.5,0.6,0.7]                       
            self.search_space['Bert_style_model'] = [ 'bert-base-uncased', 'roberta-base']    # , 'roberta-base'                                                                        
            self.search_space['bert_lr'] = [1e-5,5e-3]           
            self.search_space['graph_const'] = ['TFIDF-Sequential','TFIDF-Semantic', 'TFIDF-Syntact'] #                           
        pass     

    def get_search_space(self):           
        return self.search_space

    @staticmethod   
    def generate_action_list(cell=1):      
        action_list = []
        for i in range(cell):
            action_list += [f"self_index_{i}", "gnn"]
        action_list += ["act", "concat_type"]
        return action_list
        
class GraphclassSearchSpace(object):        
    def __init__(self, search_space=None, max_cell=10):
        if search_space:
            self.search_space = search_space
        else:
            self.search_space = {}
            self.search_space["act"] = act_list   
            self.search_space["gnn"] = gnn_list   
            self.search_space["aggr"] = ["add", "mean", "max"]           
            self.search_space['learning_rate'] = [1e-2,  5e-05, 5e-3]           
            self.search_space['dropout'] = [ 0.4, 0.5, 0.6]    
            self.search_space['weight_decay'] = [5e-5]                                          
            self.search_space['hidden_unit'] = [128, 256,64]           
            self.search_space['balance_factor'] = [0.5,0.6,0.7]                  
            self.search_space['Bert_style_model'] = [  'bert-base-uncased']  # 'roberta-base',                                    
            self.search_space['bert_lr'] = [1e-5,5e-3]
            self.search_space['graph_const'] = ['TFIDF-Sequential'] #, 'TFIDF-Semantic', 'TFIDF-Syntact'              
        pass     

    def get_search_space(self):          
        return self.search_space

    @staticmethod   
    def generate_action_list(cell=1):
        action_list = []
        for i in range(cell):
            # action_list += [f"self_index_{i}", "gnn"]
            action_list += ["gnn"]
        action_list += ["act", "aggr"]      
        return action_list      

if __name__ == "__main__":
    obj = IncrementSearchSpace()
    print(obj.generate_action_list())
    print(obj.get_search_space())
