import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from GNNs.Edge_GATConv import Edge_GATConv  
from GNNs.Edge_GCNConv import Edge_GCNConv   
from transformers import AutoModel, AutoTokenizer
from TGRL_SearchSpace.search_space import gnn_map, act_map
from transformers import logging
from GNNs.Graph_Conv import Graph_Conv          
from GNNs.CG_Conv import CG_Conv       
from GNNs.Transformer_Conv import Transformer_Conv
from GNNs.Hypergraph_Conv import Hypergraph_Conv   
# from search_space import gnn_map, act_map     
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
logging.set_verbosity_error()                       
class InDAGLayer(nn.Module):
    def __init__(self, args, embeds_pretrained,action, num_feat, num_classes, num_hidden,edge_dim=1, dropout=0.6, layers=2, stem_multiplier=2, bias=True):   
        super(InDAGLayer, self).__init__()            
        self._layers = layers        
        self.dropout = dropout    
        self.num_classes = num_classes
        num_classes1 = num_classes            
        his_dim, cur_dim, hidden_dim, out_dim, edge_dim = num_feat, num_feat, num_hidden, num_hidden, edge_dim
     
        self.his_dim = his_dim
        self.cells = nn.ModuleList()            
        for i in range(layers):               
            cell = Cell(action, his_dim, cur_dim, hidden_dim, out_dim,edge_dim, concat=False, bias=bias)         
            self.cells += [cell]       
            his_dim = cur_dim         
            cur_dim = cell.multiplier * out_dim if action[-1] == "concat" else out_dim       
        self.classifier = nn.Linear(cur_dim, self.num_classes, bias=True)               
        pass      
        self.layer_norm = args.layer_norm     
    
        # for inductive settings                        
        # print('args: ', args)                          
        # self.n_node = args.n_word       
        # self.d_model = args.d_model
        # self.n_class = args.n_class
        # self.max_len_text = args.max_len_text
        # self.layer_norm = args.layer_norm
        # self.relu = args.relu
        self.dataset = args.dataset        
        # self.mean_reduction = args.mean_reduction        

        if self.layer_norm:        
            self.ln = nn.LayerNorm(self.his_dim)          
        # self.dropout_i = nn.Dropout(args.dropout)              

        # self.weight_edge = nn.Embedding((self.n_node - 1) * self.n_node + 1, 1, padding_idx=0)  # +1 for padding
        # self.eta_node = nn.Embedding(self.n_node, 1, padding_idx=0)  # Nn, node weight for itself
        # nn.init.xavier_uniform_(self.weight_edge.weight)
        # nn.init.xavier_uniform_(self.eta_node.weight)

        # self.fc = nn.Linear(self.d_model, self.num_classes, bias=True)     

    def forward(self, x, edge_index, edge_attr, batch):            

        device = torch.device('cuda') 
        
        if self.layer_norm:
            x = self.ln(x)     
        # x = self.embedding(x)                                               
        # x = F.dropout(x, p=self.dropout, training=self.training)               
        s0 = s1 = x
        for i, cell in enumerate(self.cells):  # this determine how many layers                
            s0, s1 = s1, cell(s0, s1, edge_index,edge_attr, self.dropout)
        out = s1
        out1 = global_mean_pool(out, batch)                                                     
        out2 = global_max_pool(out, batch)                                            
        # out2 = global_add_pool(out, batch)                                                        
        out = out1 + out2
        # out = F.dropout(out, p=self.dropout, training=self.training)               
        scores = self.classifier(out)        
        if self.dataset == 'Wiki10':            
            scores = torch.sigmoid(scores)          
        return scores                   
        
        # return logits        

class Cell(nn.Module):

    def __init__(self, action_list, his_dim, cur_dim, hidden_dim, out_dim,edge_dim, concat, bias=True):
        '''

        :param action_list: like ['self_index', 'gnn'] * n +['act', 'concat_type']
        :param his_dim:
        :param cur_dim:
        :param hidden_dim:
        :param out_dim:
        :param concat:
        :param bias:
        '''
        assert hidden_dim == out_dim  # current version only support this situation   
        super(Cell, self).__init__()
        self.his_dim = his_dim
        self.cur_dim = cur_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.concat_of_multihead = concat
        self.bias = bias
        self.edge_dim = edge_dim
        self.preprocess0 = nn.Linear(his_dim, hidden_dim, bias)
        self.preprocess1 = nn.Linear(cur_dim, hidden_dim, bias)

        self._indices = []
        self._gnn = nn.ModuleList()
        self._compile(action_list)


    def _compile(self, action_list):    
        cells_info = action_list[:-2]
        assert len(cells_info) % 2 == 0

        self._steps = len(cells_info) // 2
        # print('self._steps:', self._steps)
        self.multiplier = self._steps
        self._act = act_map(action_list[-2])
        self._concat = action_list[-1]
        # print('self._concat:', self._concat)
      
        for i, action in enumerate(cells_info):    
            if i % 2 == 0:
                self._indices.append(action)  # action is a indice
                # print('self._indices:', self._indices)
            else:
                # print('action:', action)   
                self._gnn.append(gnn_map(action, self.hidden_dim, self.out_dim, self.concat_of_multihead, self.bias,self.edge_dim))

    def forward(self, s0, s1, edge_index,edge_attr, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
                   
        states = [s0, s1]           
        for i in range(self._steps):
            # print('self._indices[i]:', self._indices[i])        
            # print('states.shape: ',len(states))            
            h1 = states[self._indices[i]]
            op1 = self._gnn[i]
            # print(h1.shape, edge_index.shape,edge_attr.shape)                                                                                                                           
            # print(type(op1))                                                                                            
            
            if str(op1) == str(Edge_GATConv(1, 1,'add', 1, 1, True, True)) or str(op1) == str(Edge_GCNConv(1, 1,'add', 1, True, True)) or str(op1) == str(Graph_Conv(1, 1,'add', 1, True, True)) or str(op1) == str(CG_Conv(1, 1,'add', 1, True, True))  or str(op1) == str(Transformer_Conv(1, 1,'add', 1, True, True)) or str(op1) == str(Hypergraph_Conv(1, 1,'add', 1, True, True)):
                # print('hereerererer')                      
                s = op1(h1, edge_index,edge_attr)     

            else:                           
                s = op1(h1, edge_index)                          
            
            s = F.dropout(s, p=drop_prob, training=self.training)                           
            states += [s]
        if self._concat == "concat":
            return self._act(torch.cat(states[2:], dim=1))
        else:                   
            tmp = states[2]    
            for i in range(2,len(states)):         
                if self._concat == "add":
                    tmp = torch.add(tmp, states[i])
                elif self._concat == "product":
                    tmp = torch.mul(tmp, states[i])
                               
            return tmp


class BertModel(nn.Module):    
    def __init__(self, num_feat, num_classes,dropout, pretrained_model='roberta_base'):   
        super(BertModel, self).__init__()           
        self.dropout = dropout   
        num_classes = num_classes          
        num_classes1 = num_classes                     
        self.his_dim, self.cur_dim = num_feat, num_feat        
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)         
        self.bert_model = AutoModel.from_pretrained(pretrained_model)     
        self.cur_dim = list(self.bert_model.modules())[-2].out_features
        self.his_dim = list(self.bert_model.modules())[-2].out_features
        self.classifierbert = nn.Linear(self.cur_dim, num_classes1)    
        pass
     
    def forward(self, x,input_ids, attention_mask):                      
        # x = F.dropout(x, p=self.dropout, training=self.training)                                
        # cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]     
        bert_logit = self.classifierbert(x)                   
        
        return bert_logit       