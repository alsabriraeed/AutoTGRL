import pickle
import numpy as np
import torch
from torch.utils.data import Dataset #, DataLoader
from torch_geometric.loader import DataLoader   #    ,  DataLoaderIter
# from torch.utils.data import Dataset, DataLoader   
from itertools import chain
from torch_geometric.data import Data                  
import torch.nn as nn
import torch.nn.functional as F           
            
def construct_data( args, embeds_pretrained, data, gt, weight):
    device = torch.device('cuda')    
    length = len(gt)            
    n_word = args.n_word        
    n_degree = args.n_degree
    max_len_text = args.max_len_text
    n_node = args.n_word         
    d_model = args.d_model                    
    
    if args.pretrained:
        embedding = nn.Embedding.from_pretrained(embeds_pretrained, freeze=False, padding_idx=0)
    else:     
        embedding = nn.Embedding(n_node, d_model, padding_idx=0)
        nn.init.xavier_uniform_(embedding.weight)
    dataset = []      
    for idx in range(len(data)):           
        text_tokens = data[idx]
        len_text = len(text_tokens)                   
                             
        edgelist_f =[]     
        edgelist_b =[]         
        edge_list =[]      
        ew = []    
        for idx_token in range(len_text):
            nb_front, nb_tail = [], []      
            for i in range(n_degree):
                before_idx = idx_token - 1 - i
                nb_front.append(text_tokens[before_idx] if before_idx > -1 else 0)
                
                if before_idx > -1:
                    keyi = str(text_tokens[idx_token]) + ',' + str(text_tokens[before_idx])
                    keyj = str(text_tokens[before_idx])  + ',' + str(text_tokens[idx_token])
                    if (text_tokens[idx_token] == text_tokens[before_idx]):
                        ew.append([1])                     
                    else:
                        if keyi in weight.keys():
                            ew.append([weight[keyi]])    
                        elif keyj in weight.keys():
                            ew.append([weight[keyj]])    
                        else:
                            ew.append([0])        
                    edgelist_f.append(idx_token)       
                    edgelist_b.append(before_idx)                                
                after_idx = idx_token + 1 + i
                nb_tail.append(text_tokens[after_idx] if after_idx < len_text else 0)      
                      
                if after_idx < len_text:      
                    keyi = str(text_tokens[idx_token]) + ',' + str(text_tokens[after_idx])
                    keyj = str(text_tokens[after_idx])  + ',' + str(text_tokens[idx_token])
                    if (text_tokens[idx_token] == text_tokens[after_idx]):
                        ew.append([1])         
                    else:         
                        if keyi in weight.keys():   
                            ew.append([weight[keyi]])
                        elif keyj in weight.keys():
                            ew.append([weight[keyj]])    
                        else:
                            ew.append([0])            
                    edgelist_f.append(idx_token)                
                    edgelist_b.append(after_idx)               
        
        edge_index = [edgelist_f, edgelist_b]           
        # print('edges: ', len(edge_index[0]), 'weights: ',len(ew))                                  
        edge_index = torch.tensor(edge_index, dtype=torch.int64)                                   
        # ew1 = np.array([[1] * edge_index[0].shape[0]]).T                    
        # ew1 = torch.from_numpy(ew1)                                                  
        # ew = torch.tensor(ew1, dtype=torch.float)                             
        
        ew = np.array(ew)                      
        ew = torch.from_numpy(ew)                       
        ew = torch.tensor(ew, dtype=torch.float)                      
             
        x = torch.tensor(data[idx], dtype=torch.long)            
        x = embedding(x)
        x = x.detach()  
        y = torch.tensor(gt[idx])   
        dataset.append(Data(x=x, edge_index=edge_index,ew=ew,y=y))                   
    return dataset             
# --------------------------------------------------------------------------------------------
def construct_data1( args, embeds_pretrained, data, gt, weight, num_classes):
    device = torch.device('cuda')    
    length = len(gt)
    n_word = args.n_word
    n_degree = args.n_degree
    n_node = args.n_word         
    d_model = args.d_model                            
    dataset = []            
    for idx in range(len(data)):
        labels = np.zeros(num_classes)
        values = [1]           
        indices = gt[idx]          
        labels[indices] = values            
        text_tokens = data[idx]
        len_text = len(text_tokens)                         
                             
        edgelist_f =[]     
        edgelist_b =[]         
        edge_list =[]      
        ew = []    
        for idx_token in range(len_text):
            nb_front, nb_tail = [], []      
            for i in range(n_degree):
                before_idx = idx_token - 1 - i
                nb_front.append(text_tokens[before_idx] if before_idx > -1 else 0)
                
                if before_idx > -1:
                    edgelist_f.append(idx_token)       
                    edgelist_b.append(before_idx)                                
                after_idx = idx_token + 1 + i
                nb_tail.append(text_tokens[after_idx] if after_idx < len_text else 0)      
                      
                if after_idx < len_text:      
                    edgelist_f.append(idx_token)                
                    edgelist_b.append(after_idx)               
        
        edge_index = [edgelist_f, edgelist_b]           
        edge_index = torch.tensor(edge_index, dtype=torch.int64)                                   
        ew1 = np.array([[1] * edge_index[0].shape[0]]).T                    
        ew1 = torch.from_numpy(ew1)                                                  
        ew = torch.tensor(ew1, dtype=torch.float)                             
             
        x = torch.tensor(embeds_pretrained[idx], dtype=torch.float)                     
        y = torch.tensor(labels, dtype=torch.float)               
        dataset.append(Data(x=x, edge_index=edge_index,ew=ew,y=y))                        
    return dataset
# ------------------------------------------------------------------------------------------------------------    
def get_dataloader(args):     
    """ Get dataloader, word2idx and pretrained embeddings """     

    with open(args.path_data, 'rb') as f:
        mappings = pickle.load(f)
    # label2idx = mappings['label2idx']    
    word2idx = mappings['word2idx']
    num_word = len(word2idx)
    tr_data = mappings['tr_data']
    num_train = len(tr_data)
    tr_gt = mappings['tr_gt']
    num_classes = len(set(tr_gt))                 
    val_data = mappings['val_data']
    num_val = len(val_data)
    val_gt = mappings['val_gt']
    te_data = mappings['te_data']
    num_test = len(te_data)
    te_gt = mappings['te_gt']
    embeds = mappings['embeds']     
    weight = mappings['weight']       
    args_prepare = mappings['args']         

    if args_prepare['d_pretrained'] != args.d_model:
        raise ValueError('Experiment settings do not match data preprocess settings. '
                         'Please re-run prepare.py with correct settings.')
    args.n_class = args_prepare['n_class']                    

    args.n_word = len(word2idx)                     
    train_data = construct_data(args, torch.Tensor(embeds), tr_data, tr_gt,weight)      

    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              num_workers=args.num_worker, shuffle=False)               
                                 
    val_data = construct_data(args, torch.Tensor(embeds), val_data, val_gt,weight)      
    valid_loader = DataLoader(val_data, batch_size=args.batch_size,
                              num_workers=args.num_worker, shuffle=False)                 
    
    test_data = construct_data(args, torch.Tensor(embeds), te_data, te_gt, weight)     
    test_loader = DataLoader(test_data, batch_size=args.batch_size,
                              num_workers=args.num_worker, shuffle=False)                                

    return train_loader, valid_loader, test_loader, word2idx, torch.Tensor(embeds), num_classes, num_train, num_val, num_test, num_word 
    
# --------------------------------------------------------------------------------------------------------          
def get_dataloader1(args):     
    """ Get dataloader, word2idx and pretrained embeddings """     

    with open(args.path_data, 'rb') as f:
        mappings = pickle.load(f)
    # print(mappings.keys())                
    # label2idx = mappings['label2idx']
    num_val = int(mappings['num_train'] * 0.1 )
    num_train = int(mappings['num_train'] - num_val)
    tr_data = mappings['tr_data'][:num_train]
    val_data = mappings['tr_data'][num_train:]
    tr_gt = mappings['tr_gt'][:num_train]
    val_gt = mappings['tr_gt'][num_train:]             
    
    word2idx = mappings['word2idx']
    num_word = mappings['num_word']
     
    num_classes = len(mappings['label2idx'])                       

    te_data = mappings['te_data']    
    num_test = mappings['num_test']
    te_gt = mappings['te_gt']
    embeds_test = mappings['embeds_test']
    embeds_train = mappings['embeds_train'][:num_train]
    embeds_val = mappings['embeds_train'][num_train:]
    weight = 0
    embeds = 0     
    # weight = mappings['weight']                        
    # args_prepare = mappings['args']                            

    # if args_prepare.d_pretrained != args.d_model:          
    #     raise ValueError('Experiment settings do not match data preprocess settings. '
    #                      'Please re-run prepare.py with correct settings.')       
    # args.n_class = args_prepare.n_class                        

    args.n_word = len(word2idx)                       
    train_data = construct_data1(args, embeds_train, tr_data, tr_gt,weight,num_classes)      

    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              num_workers=args.num_worker, shuffle=False)                    
    # for batch in train_loader:      
    #     print(batch.ptr.shape[1] -1 )                          
    val_data = construct_data1(args, embeds_val, val_data, val_gt,weight,num_classes)      
    valid_loader = DataLoader(val_data, batch_size=args.batch_size,
                              num_workers=args.num_worker, shuffle=False)                             
    
    test_data = construct_data1(args, embeds_test, te_data, te_gt, weight,num_classes)           
    test_loader = DataLoader(test_data, batch_size=args.batch_size,
                              num_workers=args.num_worker, shuffle=False)                                                

    return train_loader, valid_loader, test_loader, word2idx, torch.Tensor(embeds), num_classes, num_train, num_val, num_test, num_word
