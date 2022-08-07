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
from transformers import logging as log
log.set_verbosity_error()       
from utils.utils import *
from torch_geometric.data import Data    
from transformers import AutoModel, AutoTokenizer    
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

def evaluate_text_level(output, labels):
    _, indices = torch.max(output, dim=1)
    correct = torch.sum(indices == labels)      
    # na = labels.cpu().numpy()                       
    return correct.item() #* 1.0 / len(labels) #labels.item()                                            

def evaluate_text_level_test(output, labels):
    _, indices = torch.max(output, dim=1)
    correct = torch.sum(indices == labels)      
    return correct.item() #* 1.0 / len(labels) #labels.item()
    
def encode_input(text, tokenizer,max_length):
        input = tokenizer(text, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
        return input.input_ids, input.attention_mask
# Training
def update_feature(model, data, input_ids,attention_mask, doc_mask):
    # global
    device = torch.device('cuda')
    cpu = th.device('cpu')
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
    
def load_data(args, dataset="mr", supervised=False, full_data=True):
    '''
    support semi-supervised and supervised
    :param dataset:
    :param supervised:
    :return:
    '''    
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)        
    if dataset in ["20ng","R8","ohsumed",'mr','SST-2','SST-1']:    
        all_x,all_y,x, y, adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(dataset, args.path_data)

   
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
        data = Data(x=x, edge_index=edge_index.t().contiguous(),y=y,train_mask=train_mask,val_mask=val_mask,test_mask=test_mask,edge_attr =edge_feature)

                   
    
    return data   


class Text_Classification(object):

    def __init__(self, args):
        self.args = args    
        self.early_stop_manager = EarlyStop(10)
        # instance for class that imlement top average and average
        self.reward_manager = TopAverage(10)

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

        self.param_file = args.param_file     
        self.shared_params = None
        # self.m = args.m           
        # self.bert_init = args.bert_init
        self.criterion = torch.nn.BCELoss()

        self.loss_fn = torch.nn.functional.nll_loss
        # super(MicroCitationManager, self).__init__(args)     

        if hasattr(args, "supervised"):
            self.data= load_data(self.args,args.dataset, args.supervised)

        else:   
            self.data= load_data(self.args, args.dataset)
        self.args.in_feats = self.in_feats = self.data.num_features
        self.args.in_edge_attr= self.in_edge_attr= self.data.edge_attr.size(-1)

        self.args.num_class = self.n_classes = self.data.y.max().item() +1            
        device = torch.device('cuda' if args.cuda else 'cpu')
        self.data.to(device)              
        self.corpse_file = args.path_data +'/data_for_transd_Learn/SeqGR/corpus/' + args.dataset +'_shuffle.txt'    
        
    def build_gnn(self, actions):                                          
        self.bertmodel     = BertModel(self.in_feats, self.n_classes, dropout=self.args.in_drop,pretrained_model=self.args.Bert_style_model)
        model = DAGLayer(actions, self.bertmodel.his_dim, self.n_classes, layers=self.args.layers_of_child_model, num_hidden=self.args.num_hidden,
                         edge_dim= self.in_edge_attr, dropout=self.args.in_drop,pretrained_model=self.args.Bert_style_model)
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

        self.record_action_info1(self.args.actions, reward, val_acc)

        return reward, val_acc

    def record_action_info1(self, origin_action, reward, val_acc):     
        with open(self.args.dataset + "_" + self.args.search_mode + self.args.submanager_log_file, "a") as file:
            file.write(str(origin_action))     
            file.write(";")
            file.write(str(reward))    

            file.write(";")
            file.write(str(val_acc))
            file.write("\n")
            
    def evaluate1(self, actions=None, format="two"):
        actions = process_action(actions, format, self.args)           

        # create model      
        model = self.build_gnn(actions)           

        if self.args.cuda:
            model.cuda()           

        # use optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)      
        
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
    def run_model(self,model, optimizer,args, loss_fn, data, epochs, early_stop=5, tmp_model_file="geo_citation.pkl",
                  half_stop_score=0, return_best=False, cuda=True, need_early_stop=False, show_info=False, pretrained_model='roberta-base', m=0.7):                                 
        dur = []                                                                                              
        begin_time = time.time()                                                                                                          
        best_performance = 0                       
        min_val_loss = float("inf")                                                   
        min_train_loss = float("inf")                                
        model_val_acc = 0    

        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])                   
        # print('Parameter Size: ', params)                 
        
        max_length = args.max_length
        batch_size = args.batch_size
        m = args.m   
        bert_init = args.Bert_style_model
        pretrained_bert_ckpt = '/checkpoint/'+str(args.Bert_style_model)+'_'
        # pretrained_bert_ckpt = args.pretrained_bert_ckpt      
        checkpoint_dir = args.checkpoint_dir
        gcn_model = args.gcn_model
        
        bert_lr = args.bert_lr
        pretrained_model=args.Bert_style_model
        dataset = args.dataset     
        
        if checkpoint_dir is None:
            ckpt_dir = './checkpoint/{}_{}_{}'.format(bert_init, gcn_model, dataset)
        else:
            ckpt_dir = checkpoint_dir
        os.makedirs(ckpt_dir, exist_ok=True)    
        
        # compute number of real train/val/test/word nodes and number of classes     
        nb_node = data.x.shape[0]
        nb_train, nb_val, nb_test = data.train_mask.sum(), data.val_mask.sum(), data.test_mask.sum()
        nb_word = nb_node - nb_train - nb_val - nb_test
        # print(args.num_class)                                                      
        nb_class = args.num_class   
            
        device = torch.device('cuda')           
        if pretrained_bert_ckpt is not None:                                                   
            ckpt = th.load(args.path_data+pretrained_bert_ckpt+ dataset +'/checkpoint.pth', map_location=device)      
            self.bertmodel.bert_model.load_state_dict(ckpt['bert_model'])          
            self.bertmodel.classifierbert.load_state_dict(ckpt['classifier'])

        # load documents and compute input encodings                      
        with open(self.corpse_file, 'r') as f:          
            text = f.read()
            text = text.replace('\\', '')       
            text = text.split('\n')      

        input_ids, attention_mask = encode_input(text, self.bertmodel.tokenizer,max_length)
        input_ids = th.cat([input_ids[:-nb_test], th.zeros((nb_word, max_length), dtype=th.long), input_ids[-nb_test:]])
        attention_mask = th.cat([attention_mask[:-nb_test], th.zeros((nb_word, max_length), dtype=th.long), attention_mask[-nb_test:]])
        attention_mask = attention_mask.to(device)                
        input_ids = input_ids.to(device)
        self.data.to(device)          

        # document mask used for update feature            
        optimizer1 = th.optim.Adam([
        {'params': self.bertmodel.bert_model.parameters(), 'lr': bert_lr},
        {'params': model.classifier.parameters(), 'lr': bert_lr},
    ], lr=1e-3 )   
        scheduler = lr_scheduler.MultiStepLR(optimizer1, milestones=[30], gamma=0.1)
        optimizer1.step()     
        optimizer1.zero_grad()      
        scheduler.step()

        doc_mask  = data.train_mask + data.val_mask + data.test_mask
        data.x = th.zeros((nb_node, self.bertmodel.his_dim))                      
        data = update_feature(self.bertmodel, data, input_ids,attention_mask,  doc_mask)
        # print('data.x.shape: ', data.x.shape)        
        # torch.save(data, path1+dataset+'data.pt')                                              
        th.cuda.empty_cache()                    
                                   
        #-----------------------------------------       
        for epoch in range(1, epochs + 1):
            model.train()
            t0 = time.time()
            # forward   
            
            logits = model( data.x,attention_mask,input_ids, data.edge_index, data.edge_attr)
            logits = F.log_softmax(logits, 1)                       
            
            loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])
            
            loss.backward()      
            optimizer.step()            
            optimizer.zero_grad()
            scheduler.step()        
            train_loss = loss.item()
            t1 = time.time() - t0
            t2 = time.time()      
            # evaluate                     
            model.eval()            
            logits = model(data.x,attention_mask,input_ids, data.edge_index, data.edge_attr)

            logits = F.log_softmax(logits, 1)                
                            
            train_acc = evaluate(logits, data.y, data.train_mask)       
            dur.append(time.time() - t0)              
            t3 = time.time() - t2      
            val_acc = evaluate(logits, data.y, data.val_mask)
            test_acc = evaluate(logits, data.y, data.test_mask)

            loss = loss_fn(logits[data.val_mask], data.y[data.val_mask])
            val_loss = loss.item()
            if val_loss < min_val_loss:  # and train_loss < min_train_loss
                min_val_loss = val_loss
                min_train_loss = train_loss
                model_val_acc = val_acc     
            if test_acc > best_performance:                        
                best_performance = test_acc               
            if show_info:               
                print(
                    "Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f} | acc {:.4f} | val_acc {:.4f} | test_acc {:.4f}".format(
                        epoch, loss.item(), np.mean(dur), train_acc, val_acc, test_acc))
                # print(
                #     " {:05d} \t  {:.4f}  \t {:.4f} \t  {:.4f}".format(
                #         epoch, loss.item(), t1, test_acc))                
                # print(      
                #     "Epoch  train Time(s) {:.4f} test Time(s) {:.4f} ".format(
                #         t1,t3))                          

                end_time = time.time()               
                # print("Each Epoch Cost Time: %f " % ((end_time - begin_time) / epoch))
        print(f"Classification Test Accuracy : {best_performance}")                 

        self.save_embedding(logits)        
        if return_best:
            return self.args.actions, model_val_acc, best_performance     
        else:
            return self.args.actions, model_val_acc      
        #----------------------------
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
                                                                   
            f = open(self.args.path_data  +'/data_for_transd_Learn/SeqGR/corpus/' + dataset + '_vocab.txt', 'r')
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
            f = open(self.args.path_data +'/data_for_transd_Learn/SeqGR' +'/' + dataset + '_word_vectors.txt', 'w')    
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
            f = open(self.args.path_data +'/data_for_transd_Learn/SeqGR' + '/' + dataset + '_doc_vectors.txt', 'w')           
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
    
    
    

# if acc_val > acc_best or (acc_val == acc_best and loss_best - val_loss > args.loss_eps):       
#     es_patience = 0    
#     state_best = copy.deepcopy(model.state_dict())        
#     loss_best = val_loss     
#     acc_best = acc_val      
#     epoch_best = epoch    
# else:
#     es_patience += 1
#     if es_patience >= args.es_patience_max:
#         print('\n[Warning] Early stopping model')
#         print('\t| Best | epoch {:d} | loss {:5.4f} | acc {:5.4f} |'
#               .format(epoch_best, loss_best, acc_best))   
#         break       

# # self.save_embedding(logits)                     
# # logging
# print('\t| Valid | loss {:5.4f} | acc {:5.4f} | es_patience {:.0f}/{:.0f} |'
#   .format(val_loss, acc_val, es_patience, args.es_patience_max))

# # testing phase
# print('\n[Testing]')
# model.load_state_dict(state_best)     
# if args.save_model:
#     with open(args.path_model_params, 'wb') as f:             
#         torch.save(model.state_dict(), f)
#     with open(args.path_model, 'wb') as f:      
#         torch.save(model, f)                          
