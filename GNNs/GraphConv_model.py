import torch
import numpy as np
 
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, CGConv,HypergraphConv, TransformerConv
from torch_geometric.nn import MessagePassing
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer, Engine
from ignite.metrics import Accuracy, Loss
from matplotlib.pyplot import plot as plt
from torch.nn import Linear
from sklearn.metrics import accuracy_score#, precision_score, recall_score
from tqdm import tqdm
import torch.nn.functional as F
path1 = '/content/drive/My Drive/ColabNotebooks/My_GraphNas/graphnas/'
   
mod=GraphConv 
data=torch.load(path1+'SST-2'+"data.pt")        

class GNN(MessagePassing):
    def __init__(self):
        super(GNN, self).__init__(aggr='mean')
        hdl = 2              
        self.conv1 = mod(data.num_node_features, hdl)
        self.conv2 = mod(hdl, 3)
       
    def forward(self):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x,0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = GNN().to(device), data.to(device)
optimizer = torch.optim.Adam([
    dict(params=model.conv1.parameters(), weight_decay=5e-4),
    dict(params=model.conv2.parameters(), weight_decay=5e-4)
], lr=0.001)  # Only perform weight-decay on first convolution.


def train(engine, batch):    
    model.train()
    optimizer.zero_grad()
    (idx, ) = [x.to(gpu) for x in batch]      
    train_mask = data.train_mask[idx].type(th.BoolTensor)     
    A= F.nll_loss(model()[train_mask], data.y[idx][data.train_mask])       
    
    print(f'loss ={A}')      
    A.backward()
    optimizer.step()


def test(engine, batch):
    model.eval()
    logits, accs = model(), []
    (idx, ) = [x.to(gpu) for x in batch]
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()     
        accs.append(acc)
    return accs


best_val_acc = test_acc = 0
for epoch in range(1, 201):
    # train()
    trainer = Engine(train)
    tester = Engine(test) 
    train_acc, val_acc, tmp_test_acc =  tester.state.accs      
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, best_val_acc, test_acc))

