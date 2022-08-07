import pickle    
from classification.In_DAG_Layer import InDAGLayer , BertModel    
from classification.DAG_Layer import DAGLayer          
# from In_GNNLayer import InGNNLayer, BertModel                          
import torch.utils.data as Data1                     
import torch
import torch as th         
import os.path as osp    
from torch_geometric.datasets import Planetoid, Coauthor, Amazon
import torch_geometric.transforms as T
from utils.label_split import fix_size_split
from utils.model_utils import EarlyStop, TopAverage, process_action
import torch.nn.functional as F
from sklearn.metrics import hamming_loss, accuracy_score
import time
import sys       
import os  
import copy
from tqdm import tqdm            
from torch.optim import lr_scheduler                
# path = '/content/drive/My Drive/ColabNotebooks/TextGNAS/data/'    
# path1 = '/content/drive/My Drive/ColabNotebooks/TextGNAS/data/'     
# path_indcutive = '/content/drive/My Drive/ColabNotebooks/TextGNAS/idata/'     
# checkpoint_path = '/content/drive/My Drive/ColabNotebooks/TextGNAS/'                  
from transformers import logging as log
log.set_verbosity_error()       
from utils.utils import *
from torch_geometric.data import Data    
from transformers import AutoModel, AutoTokenizer
from build_inductive_graphs.dataloader import  get_dataloader1 #get_dataloader,                
import os
from torch_geometric.loader import DataLoader
from itertools import cycle
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def evaluate(output, labels, mask):    
    _, indices = torch.max(output, dim=1)
    correct = torch.sum(indices[mask] == labels[mask])
    # print(mask.sum().item())               
    return correct.item() * 1.0 / mask.sum().item()     
import torch.nn as nn            
def evaluate_text_level(output, labels):
    _, indices = torch.max(output, dim=1)
    # print('indices: ', indices)     
    # print('labels: ', labels)           
    correct = torch.sum(indices == labels)      
    # na = labels.cpu().numpy()                       
    return correct.item() #* 1.0 / len(labels) #labels.item()                                            

def evaluate_text_level_test(output, labels):
    _, indices = torch.max(output, dim=1)
    # print('indices: ', indices)      
    # print('labels: ', labels)               
    correct = torch.sum(indices == labels)      
    # na = labels.cpu().numpy()                       
    return correct.item() #* 1.0 / len(labels) #labels.item()
    
def encode_input(text, tokenizer,max_length):
        input = tokenizer(text, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
    #     print(input.keys())   
        return input.input_ids, input.attention_mask
# Training
def update_feature(model, data, input_ids,attention_mask, doc_mask):
    # global
    device = torch.device('cuda')
    cpu = th.device('cpu')
    # gpu = th.device('cuda:0')      
    # no gradient needed, uses a large batchsize to speed up the process
    dataloader = Data1.DataLoader(
        Data1.TensorDataset(input_ids[doc_mask], attention_mask[doc_mask]),
        batch_size=1024
    )      
    with th.no_grad():
        model = model.to(device)
        model.eval()
        cls_list = []
        for i, batch in enumerate(dataloader):   
            input_ids, attention_mask = [x.to(device) for x in batch]
            output = model.bert_model(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0]
            cls_list.append(output.cpu())
        cls_feat = th.cat(cls_list, axis=0).to(device)    
    data = data.to(device)                     
    data.x[doc_mask] = cls_feat
    return data

def update_feature_inductive(model, input_ids,attention_mask, doc_mask):
    # global
    device = torch.device('cuda')
    cpu = th.device('cpu')
    # gpu = th.device('cuda:0')      
    # no gradient needed, uses a large batchsize to speed up the process
    dataloader = Data1.DataLoader(
        Data1.TensorDataset(input_ids[doc_mask], attention_mask[doc_mask]),
        batch_size=512     
    )           
    with th.no_grad():
        model = model.to(device)
        model.eval()
        cls_list = []
        for i, batch in enumerate(dataloader):   
            input_ids, attention_mask = [x.to(device) for x in batch]
            output = model.bert_model(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0]
            cls_list.append(output.cpu())
        cls_feat = th.cat(cls_list, axis=0).to(device)    
    cls_feat = cls_feat.to(device)                     
    # data.x[doc_mask] = cls_feat
    return cls_feat     
   
def construct_data( args, embeds_pretrained, data, gt, weight, bert_feat, bertmodel):    
    device = torch.device('cuda')    
    length = len(gt)     
    n_word = args.n_word
    n_degree = args.n_degree
    max_len_text = args.max_len_text
    n_node = args.n_word                  
    d_model = args.d_model                         
    
    if args.dataset == 'Wiki10':
        labels = np.zeros(num_classes)
        values = [1]           
        indices = gt[idx]          
        labels[indices] = values
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
        edge_index = torch.tensor(edge_index, dtype=torch.int64)                                   
        
        ew = np.array(ew)                      
        ew = torch.from_numpy(ew)                       
        ew = torch.tensor(ew, dtype=torch.float)                      
             
        x = torch.tensor(data[idx], dtype=torch.long)            
        x = embedding(x)     
        x = x.detach()      
        x = x.to(device)
        x_bert_feat = bert_feat[idx]           
        x_bert_feat = x_bert_feat.detach()                
        x = x * x_bert_feat     
        x = x.detach()
        if args.dataset == 'Wiki10':
            y = torch.tensor(labels, dtype=torch.float)     
        else:
            y = torch.tensor(gt[idx])               
        dataset.append(Data(x=x, edge_index=edge_index,ew=ew,y=y))                   
    return dataset

def get_dataloader(args):     
    """ Get dataloader, word2idx and pretrained embeddings """          

    with open(args.path_data+'/data_for_induct_Learn/SeqGR/'+ args.dataset + '.pkl', 'rb') as f:
        mappings = pickle.load(f)        
    # label2idx = mappings['label2idx']         
    word2idx = mappings['word2idx']
    num_word = len(word2idx)
    tr_data = mappings['tr_data']
    num_train = len(tr_data)
    tr_gt = mappings['tr_gt']
    num_classes = len(set(tr_gt)) + 1                          
    val_data = mappings['val_data']
    num_val = len(val_data)
    val_gt = mappings['val_gt']
    te_data = mappings['te_data']
    num_test = len(te_data)
    te_gt = mappings['te_gt']
    embeds = mappings['embeds']     
    weight = mappings['weight']       
    args_prepare = mappings['args']
    
    
    in_feats = in_feats =  embeds.shape[1]
    max_length = args.max_length
    device = torch.device('cuda')      

    if args_prepare['d_pretrained'] != args.d_model:
        raise ValueError('Experiment settings do not match data preprocess settings. '
                         'Please re-run prepare.py with correct settings.')
    args.n_class = args_prepare['n_class']                      

    args.n_word = len(word2idx)                                  
    
    corpse_file = args.path_data + '/data_for_induct_Learn/SeqGR' +'/corpus/'+args.dataset+'.txt'
    checkpoint_path =   args.path_data                         
    with open(corpse_file, 'r') as f:                    
        text = f.read()
        text = text.replace('\\', '')                                     
        text = text.split('\n')             
    bertmodel     = BertModel(in_feats, num_classes, dropout=args.in_drop,pretrained_model=args.Bert_style_model)
    
    pretrained_bert_ckpt = '/checkpoint/'+str(args.Bert_style_model)+'_'
    if pretrained_bert_ckpt is not None:                                                                  
        ckpt = th.load(checkpoint_path+pretrained_bert_ckpt+ args.dataset +'/checkpoint.pth', map_location=device)      
        bertmodel.bert_model.load_state_dict(ckpt['bert_model'])                              
        bertmodel.classifierbert.load_state_dict(ckpt['classifier'])         
    input_ids, attention_mask = encode_input(text, bertmodel.tokenizer,max_length)            
    # input_ids = th.cat([input_ids[:-nb_test], th.zeros((nb_word, max_length), dtype=th.long), input_ids[-nb_test:]])                
    # attention_mask = th.cat([attention_mask[:-nb_test], th.zeros((nb_word, max_length), dtype=th.long), attention_mask[-nb_test:]])               
    input_ids = th.cat([input_ids[:-num_test], input_ids[-num_test:]])
    attention_mask = th.cat([attention_mask[:-num_test],  attention_mask[-num_test:]])
    attention_mask = attention_mask.to(device)                                             
    input_ids = input_ids.to(device)                           

    doc_mask  = np.array([1]*(num_train + num_val + num_test))      
    bert_features = update_feature_inductive(bertmodel, input_ids,attention_mask,  doc_mask)

            
    linear = nn.Linear(bertmodel.his_dim, in_feats, bias=True)                  
    linear = linear.to(device)       
    
    train_bert_feat = bert_features[:num_train]
    train_bert_feat = train_bert_feat.to(device)
    train_bert_feat = linear(train_bert_feat)
    train_bert_input_ids = input_ids[:num_train]
    train_bert_attention_mask = attention_mask[:num_train]     

    val_bert_feat = bert_features[num_train:num_train+num_val]
    val_bert_feat = val_bert_feat.to(device)
    val_bert_feat = linear(val_bert_feat)
    
    val_bert_input_ids = input_ids[num_train:num_train+num_val]
    val_bert_attention_masks = attention_mask[num_train:num_train+num_val]

    test_bert_feat = bert_features[num_train+num_val:]      
    test_bert_feat = test_bert_feat.to(device)            
    test_bert_feat = linear(test_bert_feat)           
    test_bert_input_ids = input_ids[num_train+num_val:]             
    test_bert_attention_mask = attention_mask[num_train+num_val:]       
    
    train_data = construct_data(args, torch.Tensor(embeds), tr_data, tr_gt,weight, train_bert_feat,bertmodel)      

    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              num_workers=args.num_worker, shuffle=False)               
    # for batch in train_loader:         
    #     print(batch)                         
    val_data = construct_data(args, torch.Tensor(embeds), val_data, val_gt,weight, val_bert_feat,bertmodel)          
    valid_loader = DataLoader(val_data, batch_size=args.batch_size,
                              num_workers=args.num_worker, shuffle=False)                 
    
    test_data = construct_data(args, torch.Tensor(embeds), te_data, te_gt, weight,test_bert_feat,bertmodel)          
    test_loader = DataLoader(test_data, batch_size=args.batch_size,
                              num_workers=args.num_worker, shuffle=False)
                              
    train_bert_feat_loader = DataLoader(train_bert_feat, batch_size=args.batch_size,
                              shuffle=False)
    val_bert_feat_loader = DataLoader(val_bert_feat, batch_size=args.batch_size,
                      shuffle=False)
    test_bert_feat_loader = DataLoader(test_bert_feat, batch_size=args.batch_size,
                      shuffle=False)
    
    train_bert_input_ids_loader = DataLoader(train_bert_input_ids, batch_size=args.batch_size,
                      shuffle=False)      
    val_bert_input_ids_loader = DataLoader(val_bert_input_ids, batch_size=args.batch_size,
                      shuffle=False)       
    test_bert_input_ids_loader = DataLoader(test_bert_input_ids, batch_size=args.batch_size,
                      shuffle=False)
                       
    train_bert_attention_mask_loader = DataLoader(train_bert_attention_mask, batch_size=args.batch_size,
                      shuffle=False)      
    val_bert_attention_mask_loader = DataLoader(val_bert_attention_masks, batch_size=args.batch_size,
                      shuffle=False)       
    test_bert_attention_mask_loader = DataLoader(test_bert_attention_mask, batch_size=args.batch_size,
                              shuffle=False)
    bert_data  = [train_bert_feat_loader,val_bert_feat_loader, test_bert_feat_loader, train_bert_input_ids_loader, val_bert_input_ids_loader, test_bert_input_ids_loader, train_bert_attention_mask_loader, val_bert_attention_mask_loader, test_bert_attention_mask_loader ]
    return train_loader, valid_loader, test_loader, word2idx, torch.Tensor(embeds), num_classes, num_train, num_val, num_test, num_word, bert_data, bertmodel
    
def load_data(args, dataset="mr", supervised=False, full_data=True):    
    '''
    support semi-supervised and supervised
    :param dataset:
    :param supervised:
    :return:
    '''    
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)              
    if dataset in ["20ng","R8","ohsumed","mr",'SST-2',"R52", 'SST-1']:   
        all_x,all_y,x, y, adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(dataset)

   
        features = features.toarray()               

        all_y = all_y.toarray()
        labels= [0 if max(i)==0 else list(i).index(max(i))+1 for i in all_y]

        edge_index = [preprocess_adj(adj)]
        edge_feature = np.array( [edge_index[0][1]]).T
                                                            
        edge_index = torch.tensor(edge_index[0][0], dtype=torch.long)
        x = torch.tensor(features, dtype=torch.float)
        y = torch.tensor(labels)
        train_mask = torch.tensor(train_mask)
        val_mask = torch.tensor(val_mask)
        test_mask = torch.tensor(test_mask)
        edge_feature = torch.tensor(edge_feature, dtype=torch.float)
        # print(edge_feature.shape,edge_index.shape)                        
       
        data = Data(x=x, edge_index=edge_index.t().contiguous(),y=y,train_mask=train_mask,val_mask=val_mask,test_mask=test_mask,edge_attr =edge_feature)


    return data   

class Text_Classification(object):

    def __init__(self, args):
        # super(Text_Classification, self).__init__(args)
        self.args = args

        self.early_stop_manager = EarlyStop(10)          
        # instance for class that imlement top average and average
        self.reward_manager = TopAverage(10)
        self.path_indcutive  = self.args.path_data + '/data_for_induct_Learn/SeqGR/'
        # self.args = args
        self.drop_out = args.in_drop
        self.multi_label = args.multi_label
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.retrain_epochs = args.retrain_epochs
        self.loss_fn = torch.nn.BCELoss()
        self.epochs = args.epochs
        self.train_graph_index = 0
        self.train_set_length = 10
        self.device = torch.device('cuda')
        self.param_file = args.param_file     
        self.shared_params = None
        # self.m = args.m           
        # self.bert_init = args.bert_init   
        self.criterion = torch.nn.BCELoss()

        self.loss_fn = torch.nn.functional.nll_loss     
        self.checkpoint_dir = args.checkpoint_dir
        # self.pretrained_bert_ckpt = 'checkpoint/'+str(self.args.Bert_style_model)+'_'
        self.max_length = self.args.max_length
        # super(MicroCitationManager, self).__init__(args)     

        
        # self.args.path_data1 = self.path_indcutive + args.dataset + '.pkl'                               
        self.train_loader, self.valid_loader, self.test_loader, self.word2idx, self.embeds_pretrained, self.num_classes, self.num_train, self.num_val, self.num_test, self.num_word,self.bert_data, self.bertmodel= get_dataloader(args)          
        self.data =[self.train_loader, self.valid_loader, self.test_loader, self.word2idx, self.embeds_pretrained]          
        self.args.in_feats = self.in_feats =  self.embeds_pretrained.shape[1]   # self.bertmodel.his_dim                            
        self.args.num_class = self.n_classes = self.num_classes + 1                                    
        # print(self.num_classes)                      
        self.args.in_edge_attr= self.in_edge_attr= 1 #self.data.edge_attr.size(-1)                                           
                      
        self.train_bert_feat_loader = self.bert_data[0]
        self.val_bert_feat_loader = self.bert_data[1]        
        self.test_bert_feat_loader = self.bert_data[2]      
        self.train_bert_input_ids_loader =self.bert_data[3]
        self.val_bert_input_ids_loader = self.bert_data[4]
        self.test_bert_input_ids_loader = self.bert_data[5]
        self.train_bert_attention_mask_loader = self.bert_data[6]
        self.val_bert_attention_mask_loader= self.bert_data[7]       
        self.test_bert_attention_mask_loader = self.bert_data[8]
        th.cuda.empty_cache()
        bert_lr = self.args.bert_lr
        pretrained_model=args.Bert_style_model

      
    def build_gnn(self, actions):               
        model = InDAGLayer(self.args, self.embeds_pretrained, actions, self.in_feats, self.n_classes, layers=self.args.layers_of_child_model, num_hidden=self.args.num_hidden,
                     edge_dim= self.in_edge_attr, dropout=self.args.in_drop)
        # self.bertmodel     = BertModel(self.in_feats, self.n_classes, num_hidden=self.args.num_hidden, dropout=self.args.in_drop,pretrained_model=self.args.Bert_style_model)        
        return model                                                

    def train(self, actions=None, format="micro"):                                                                             
        self.current_action = actions                                                                                                                                     
        print(actions)                                                            
        model_actions = actions['action']                                                          
        param = actions['hyper_param']                                      
        self.args.lr = param[0]                          
        self.args.in_drop = param[1]    
        self.args.weight_decay = param[2]
        self.args.num_hidden = param[3]
        self.args.Bert_style_model = param[4]
        self.args.bert_lr = param[5]
        self.args.actions = actions     
        return self.train1(model_actions, format=format)    

    def record_action_info(self, origin_action, reward, val_acc):
        return self.record_action_info(self.current_action, reward, val_acc)
     
    def evaluate(self, actions=None, format="micro"):
        print(actions)               
        model_actions = actions['action']
        param = actions['hyper_param']
        self.args.lr = param[0]
        self.args.in_drop = param[1]
        self.args.weight_decay = param[2]
        self.args.num_hidden = param[3]     
        self.args.Bert_style_model = param[4]    
        self.args.bert_lr = param[5]
        self.args.actions = actions  
        return self.evaluate1(model_actions, format=format)

          
    def train1(self, actions=None, format="two"):
        origin_action = actions
        actions = process_action(actions, format, self.args)
        # print("train action:", actions)

        # create model
        model = self.build_gnn(actions)

        try:
            if self.args.cuda:    
                model.cuda()
            # use optimizer             
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            model, val_acc = self.run_model(self,model, optimizer,self.args, self.loss_fn, self.data, self.epochs, cuda=self.args.cuda,
                                            half_stop_score=max(self.reward_manager.get_top_average() * 0.7, 0.4),pretrained_model=self.args.Bert_style_model)          
            val_acc = val_acc                  
        except RuntimeError as e:
            if "cuda" in str(e) or "CUDA" in str(e):
                print(e)
                val_acc = 0
            else:
                raise e
        reward = self.reward_manager.get_reward(val_acc)
        self.save_param(model, update_all=(reward > 0))    

        # self.record_action_info1(origin_action, reward, val_acc)              
        self.record_action_info1(self.args.actions, reward, val_acc)

        return reward, val_acc

    def record_action_info1(self, origin_action, reward, val_acc):     
        with open(self.args.dataset + "_" + self.args.search_mode + self.args.submanager_log_file, "a") as file:
            # with open(f'{self.args.dataset}_{self.args.search_mode}_{self.args.format}_manager_result.txt', "a") as file:
            file.write(str(origin_action))     
            file.write(";")
            file.write(str(reward))    

            file.write(";")
            file.write(str(val_acc))
            file.write("\n")
            
    def evaluate1(self, actions=None, format="two"):
        actions = process_action(actions, format, self.args)           
        # print("train action:", actions)                     

        # create model      
        model = self.build_gnn(actions)           

        if self.args.cuda:
            model.cuda()           

        # use optimizer
        
        optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr= self.args.lr,#self.args.lr  1e-3    
                                 weight_decay=1e-4)
        try:
            model, val_acc, test_acc = self.run_model(self,model, optimizer,self.args, self.loss_fn, self.data, self.epochs,
                                                      cuda=self.args.cuda, return_best=True,
                                                      half_stop_score=max(self.reward_manager.get_top_average() * 0.7,
                                                                          0.4),pretrained_model=self.args.Bert_style_model)                    
        except RuntimeError as e:
            if "cuda" in str(e) or "CUDA" in str(e):        
                print(e)
                val_acc = 0      
                test_acc = 0
            else:
                raise e      
        return val_acc, test_acc                        
    @staticmethod
    def run_model(self,model, optimizer,args, loss_fn, data, epochs, early_stop=5, tmp_model_file="R8.pkl",
                  half_stop_score=0, return_best=False, cuda=True, need_early_stop=False, show_info=False, pretrained_model='roberta-base', m=0.7):                                 
        dur = []                                                                                                    
        begin_time = time.time()                                                                                                                            
        best_performance = 0                               
        min_val_loss = float("inf")                                         
        min_train_loss = float("inf")                         
        model_val_acc = 0                 
        # print('Parameter Size: ', sum(p.numel() for p in model.parameters() if p.requires_grad))    
        
        batch_size = args.batch_size
        m = args.m   
        bert_init = args.Bert_style_model

        gcn_model = args.gcn_model
    
        
        nb_train, nb_val, nb_test = self.num_train, self.num_val, self.num_test
        nb_word = self.num_word         
        nb_class = args.num_class          
            
        device = torch.device('cuda')

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_step, gamma=self.args.lr_gamma)                      
        train_loader = self.data[0]     
        valid_loader  = self.data[1]
        test_loader   =  self.data[2]
        word2idx      =   self.data[3]       
        self.loss_fn = torch.nn.BCELoss()
        
        loss_best = 1e5
        acc_best = 0      
        epoch_best = 0   
        es_patience = 0    
        t0 = time.time()                 
        for epoch in range(1, args.epochs+1): 
            t_start = time.time()                       
            # ---------------------------------------------------Training-------------------------------------    
            model.train()        
            loss_total = 0.
            n_sample = 0
            batches =0     
            correct_pred_total = 0
            device = torch.device('cuda')
            scores_list = []
            Y_list = []       
            for i, batch in enumerate(train_loader):
                batch = batch.to(device)        
                scores_batch = model(x=batch.x, edge_index= batch.edge_index, edge_attr= batch.ew, batch=batch.batch )
                       
                optimizer.zero_grad()                                  
                loss_batch = F.cross_entropy(scores_batch, batch.y)      
                loss_batch.backward()       
                optimizer.step()                
                # # calculate loss                                     
                loss_total += loss_batch * scores_batch.shape[0]               
                n_sample += scores_batch.shape[0]
                correct_pred_total+= evaluate_text_level(scores_batch, batch.y)
            loss_train = loss_total / n_sample                     
            acc_train = correct_pred_total / n_sample         
            # ----------------------------------------------------------------------------------------------------------         
            scheduler.step()                   
            # -----------------------------------------------------evaluation-------------------------------------------      
            model.eval()    
            loss_total = 0.
            n_sample = 0
            correct_pred_total = 0
            batches =0    
            device = torch.device('cuda')    
            scores_list = []               
            with torch.no_grad():
                for i, batch in enumerate(valid_loader):
                    batch = batch.to(device)
                    scores_batch = model(x=batch.x, edge_index= batch.edge_index, edge_attr= batch.ew, batch=batch.batch )
                    loss_batch = F.cross_entropy(scores_batch, batch.y)
                    # # calculate loss            
                    loss_total += loss_batch * scores_batch.shape[0]    
                    n_sample += scores_batch.shape[0]
                    correct_pred_total += evaluate_text_level_test(scores_batch, batch.y)                 
                val_loss = loss_total / n_sample         
                acc_val = correct_pred_total / n_sample
            dur.append(time.time() - t0)
            t_end = time.time() - t_start
            # --------------------------------------------------------Testing--------------------------------------------        
            loss_total = 0.
            n_sample = 0             
            correct_pred_total = 0      
            device = torch.device('cuda')                
            scores_list = []    
            with torch.no_grad():        
                for i, batch in enumerate(test_loader):           
                    batch = batch.to(device)
                    scores_batch = model(x=batch.x, edge_index= batch.edge_index, edge_attr= batch.ew, batch=batch.batch)         

                    loss_batch = F.cross_entropy(scores_batch, batch.y)            
                    # calculate loss               
                    loss_total += loss_batch * scores_batch.shape[0]        
                    n_sample += scores_batch.shape[0]
                    correct_pred_total += evaluate_text_level_test(scores_batch, batch.y)        
                loss_test = loss_total / n_sample         
                acc_test = correct_pred_total / n_sample       
            # ---------------------------------------------------Output--------------------------------------------------------

            if loss_test < min_val_loss:  # and train_loss < min_train_loss        
                    min_val_loss = loss_test
                    min_train_loss = loss_train
                    model_val_acc = acc_val
            if acc_test > best_performance:                
                best_performance = acc_test                       
            if show_info:             
                # print(
                #     "Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f} | acc {:.4f} | acc_val {:.4f} | test_acc {:.4f}".format(
                #         epoch, loss_train.item(), t_end, acc_train, acc_val, acc_test))
                print(
                    " {:05d} \t  {:.4f} \t \t {:.4f}  \t  {:.4f}".format(
                        epoch, loss_train.item(), t_end, acc_test))                
            if acc_val > acc_best or (acc_val == acc_best and loss_best - val_loss > args.loss_eps):       
                es_patience = 0                  
                state_best = copy.deepcopy(model.state_dict())                           
                loss_best = val_loss                 
                acc_best = acc_val                     
                epoch_best = epoch                                      
            else:
                es_patience += 1
                if es_patience >= args.es_patience_max:    
                    print('\n[Warning] Early stopping model')        
                    print('\t| Best | epoch {:d} | loss {:5.4f} | acc {:5.4f} |'
                          .format(epoch_best, loss_best, best_performance))                    
                    break                               
        # self.save_embedding(logits)                                                                                              
        # logging                                                                    
                end_time = time.time()          
        print(f"Classification Accuracy : {best_performance}")                                     

        # self.save_embedding(logits)                         
        if return_best:    
            return self.args.actions, model_val_acc, best_performance    
        else:
            return self.args.actions, model_val_acc                       
        
            
    def save_embedding(self, outs):     
        # doc and word embeddings                                
        if self.args.model_type =='transductive':
            train_size = len(self.data.train_mask[self.data.train_mask==1]) + len(self.data.val_mask[self.data.val_mask==1])
            test_size = len(self.data.test_mask[self.data.test_mask==1])
            samples = self.data.x.shape[0]             
            dataset = self.args.dataset
            
            word_embeddings = outs[train_size: samples - test_size].cpu().detach().numpy()     
            train_doc_embeddings = outs[:train_size].cpu().detach().numpy()  # include val docs               
            test_doc_embeddings = outs[samples - test_size:].cpu().detach().numpy()
                                                                   
            f = open(self.args.path_data + '/data_for_induct_Learn/SeqGR/' +'/corpus/' + dataset + '_vocab.txt', 'r')       
            words = f.readlines()          
            f.close()               
            
            vocab_size = len(words)
            word_vectors = []   
            for i in range(vocab_size):
                word = words[i].strip()
                word_vector = word_embeddings[i]
                word_vector_str = ' '.join([str(x) for x in word_vector])
                word_vectors.append(word + ' ' + word_vector_str)
            
            word_embeddings_str = '\n'.join(word_vectors)
            f = open(self.args.path_data + '/data_for_induct_Learn/SeqGR/'+'/' + dataset + '_word_vectors.txt', 'w')    
            f.write(word_embeddings_str)
            f.close()
            doc_vectors = []
            doc_id = 0
            for i in range(train_size):
                doc_vector = train_doc_embeddings[i]
                doc_vector_str = ' '.join([str(x) for x in doc_vector])
                doc_vectors.append('doc_' + str(doc_id) + ' ' + doc_vector_str)
                doc_id += 1
            for i in range(test_size):      
                doc_vector = test_doc_embeddings[i]
                doc_vector_str = ' '.join([str(x) for x in doc_vector])
                doc_vectors.append('doc_' + str(doc_id) + ' ' + doc_vector_str)
                doc_id += 1                 
                  
            doc_embeddings_str = '\n'.join(doc_vectors)        
            f = open(self.args.path_data + '/data_for_induct_Learn/SeqGR/'+ '/' + dataset + '_doc_vectors.txt', 'w')           
            f.write(doc_embeddings_str)     
            f.close()
        else:
            train_size = self.num_train  + self.num_val
            test_size = self.num_test         
        
    def test_with_param(self, actions=None, format="two", with_retrain=False):
        return self.train(actions, format)
    
    def retrain(self, actions, format="two"):
        return self.train(actions, format)
    def load_param(self):
        # don't share param
        pass

    def save_param(self, model, update_all=False):
        # don't share param                      
        pass