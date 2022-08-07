"""Entry point."""
import argparse
import time

import torch          
import sys        
import search_algrithm.trainer as trainer     
import utils.tensor_utils as utils


def build_args():
    parser = argparse.ArgumentParser(description='AutoTGRL')
    register_default_args(parser)
    args = parser.parse_args()

    return args

def register_default_args(parser):
    #---------------------------------------
    parser.add_argument('--max_length', type=int, default=128, help='the input length for bert')   
    # parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gcn_model', type=str, default='gcn', choices=['gcn', 'gat'])                 
    parser.add_argument('-m', '--m', type=float, default=0.7, help='the factor balancing BERT and GNN prediction')  #0.7                                 
    parser.add_argument('--nb_epochs', type=int, default=10)         
    parser.add_argument('--bert_init', type=str, default='roberta-base',     
                        choices=['roberta-base', 'roberta-large', 'bert-base-uncased', 'bert-large-uncased'])  # update  this                      
    parser.add_argument('--pretrained_bert_ckpt', default='checkpoint/roberta-base_')  # update also this       
    parser.add_argument('--checkpoint_dir', default=None, help='checkpoint directory, [bert_init]_[gcn_model]_[dataset] if not specified')
#------------------------------------------------------------------                                                            
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'derive'],
                        help='train: Training GraphNAS, derive: Deriving Architectures')                 
    parser.add_argument('--model_type', type=str, default='transductive',
                        choices=['transductive', 'inductive'],
                        help='inductive: each docunemt a graph, transductive: a graph for the whole corpus')            
    parser.add_argument('--random_seed', type=int, default=123)                                                                                        
    parser.add_argument("--cuda", type=bool, default=True, required=False,
                        help="run in cuda mode")                                           
    parser.add_argument('--save_epoch', type=int, default=2)                                                      
    parser.add_argument('--max_save_num', type=int, default=5)                                                      
    # controller    
    parser.add_argument('--layers_of_child_model', type=int, default=2)      
    parser.add_argument('--num_of_cell', type=int, default=2)              
    parser.add_argument('--shared_initial_step', type=int, default=0)                                             
    parser.add_argument('--batch_size', type=int, default= 64)  #64  100                                                                                     
    parser.add_argument("--need_early_stop", type=bool, default=False, required=False,
                        help="need_early_stop")            
    parser.add_argument("--direct", type=bool, default=False, required=False,
                        help="calling the model directly")                              
    parser.add_argument('--entropy_mode', type=str, default='reward', choices=['reward', 'regularizer'])
    parser.add_argument('--entropy_coeff', type=float, default=1e-4)                   
    parser.add_argument('--shared_rnn_max_length', type=int, default=35)         
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--search_mode', type=str, default='micro')             
    parser.add_argument('--format', type=str, default='two')      
    parser.add_argument('--max_epoch', type=int, default=10) # 10                                     
    parser.add_argument('--ema_baseline_decay', type=float, default=0.95)
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--controller_max_step', type=int, default=100, #100                                
                        help='step for controller parameters')
    parser.add_argument('--controller_optim', type=str, default='adam')     
    parser.add_argument('--controller_lr', type=float, default=3.5e-4,
                        help="will be ignored if --controller_lr_cosine=True")
    parser.add_argument('--controller_grad_clip', type=float, default=0)
    parser.add_argument('--tanh_c', type=float, default=2.5)
    parser.add_argument('--softmax_temperature', type=float, default=5.0)
    parser.add_argument('--derive_num_sample', type=int, default=100)#100          
    parser.add_argument('--derive_finally', type=bool, default=True)
    parser.add_argument('--derive_from_history', type=bool, default=True)

    # child model   
    parser.add_argument("--dataset", type=str, default="mr", required=False,
                        help="The input dataset.")
    parser.add_argument("--epochs", type=int, default=300, #300                                                                                                                         
                        help="number of training epochs")                    
    parser.add_argument("--retrain_epochs", type=int, default=300,  #300                                                                       
                        help="number of training epochs")     
    parser.add_argument("--multi_label", type=bool, default=False,
                        help="multi_label or single_label task")
    parser.add_argument("--residual", action="store_false",
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default= 0.4, #0.6                            
                        help="input feature dropout")   
    parser.add_argument("--lr", type=float, default=0.01, 
                        help="learning rate") #0.005
    parser.add_argument("--param_file", type=str, default="r8_test.pkl",
                        help="learning rate")    
    parser.add_argument("--optim_file", type=str, default="opt_r8_test.pkl",
                        help="optimizer save path")
    parser.add_argument('--weight_decay', type=float, default=5e-4) #5e-4                 
    parser.add_argument('--max_param', type=float, default=5E6)            
    parser.add_argument('--supervised', type=bool, default=True)
    parser.add_argument('--submanager_log_file', type=str, default=f"sub_manager_logger_file_{time.time()}.txt")
    
    # inductive training 
    parser.add_argument('--mean_reduction', action='store_true', help='ablation: use mean reduction instead of max')                
    parser.add_argument('--pretrained', default= True , action='store_true', help='ablation: use pretrained GloVe')            
    parser.add_argument('--d_model', type=int, default=300, help='node representation dimensions including embedding')            
    parser.add_argument('--max_len_text', type=int, default=100,
                        help='maximum length of text, default 100, and 150 for ohsumed')                        
    parser.add_argument('--n_degree', type=int, default=11, help='neighbor region radius')#11                                                      
    parser.add_argument('--num_worker', type=int, default=0, help='number of dataloader worker')                                           
    parser.add_argument('--es_patience_max', type=int, default=30, help='max early stopped patience')                                            
    parser.add_argument('--loss_eps', type=float, default=1e-4, help='minimum loss change threshold')                         
    parser.add_argument('--seed', type=int, default=1111, help='random seed')                  
    # should be removed 
    parser.add_argument('--dropout', type=float, default=0, help='dropout rate applied to layers (0 = no dropout)')    
    parser.add_argument('--relu', action='store_true', help='ablation: use relu before softmax')
    parser.add_argument('--device', type=str, default='cuda:0', help='device for computing')            
    parser.add_argument('--path_data', type=str, default=sys.path[0], help='path of the data corpus')      
    parser.add_argument('--path_log', type=str, default='/content/drive/MyDrive/ColabNotebooks/TextGNAS/idata/result/logs/', help='path of the training logs')
    parser.add_argument('--path_model', type=str, default='/content/drive/MyDrive/ColabNotebooks/TextGNAS/idata/result/models/', help='path of the trained model')
    parser.add_argument('--lr_step', type=int, default=5, help='number of epoch for each lr downgrade')         
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='strength of lr downgrade')             
    parser.add_argument('--save_model', type=bool, default=False, help='save model for further use')              
    parser.add_argument('--layer_norm', default=True, help='ablation: use layer normalization')    
    parser.add_argument('--bert_lr', type=float, default=1e-4)                                                                                                                             
    parser.add_argument('--Bert_style_model', type=str, default='bert-base-uncased',
                    choices=['roberta-base', 'roberta-large', 'bert-base-uncased', 'bert-large-uncased'])
    

def main(args):  # pylint:disable=redefined-outer-name    

    if args.cuda and not torch.cuda.is_available():  # cuda is not available   
        args.cuda = False
    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)      
    utils.makedirs(args.dataset)                 
    trnr = trainer.Trainer(args)               
    if args.mode == 'train':
        print(args)
        trnr.train()
    elif args.mode == 'derive':
        trnr.derive()
    else:
        raise Exception(f"[!] Mode not found: {args.mode}")


if __name__ == "__main__":
    args = build_args()
    main(args)
