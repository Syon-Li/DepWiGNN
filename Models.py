

from torch import nn, zeros
import copy
import torch
import random
import os
import numpy as np
from transformers import AutoModel
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, GraphConv, GCN2Conv




def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.




class Baseline_model_stepgame(nn.Module):
    def __init__(self, device, model_name, seed_val=42,
                 embedding_size=768, num_classes=9):
        super(Baseline_model_stepgame, self).__init__()
        seed_torch(seed_val)
        self.PLM = AutoModel.from_pretrained(model_name)
        self.device = device
        self.classifier = nn.Linear(embedding_size, num_classes)
        self.dropout = nn.Dropout(p=0.2)

    
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.PLM(input_ids, attention_mask, token_type_ids)
        # last_hidden_states = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        x = self.dropout(pooler_output)
        x = self.classifier(x)
        return x
    
    
    
    
class Baseline_model_spartun(nn.Module):
    def __init__(self, device, model_name, Q_type, seed_val=42,
                 embedding_size=768, num_classes=2):
        super(Baseline_model_spartun, self).__init__()
        seed_torch(seed_val)
        self.PLM = AutoModel.from_pretrained(model_name)
        self.device = device
        self.Q_type = Q_type
        if Q_type ==  'YN':
            self.classifier = nn.Linear(embedding_size, num_classes)
        else:
            self.classifier = nn.ModuleList(nn.Linear(embedding_size, num_classes) for i in range(15))
        self.dropout = nn.Dropout(p=0.2)
    
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.PLM(input_ids, attention_mask, token_type_ids)
        # last_hidden_states = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        x = self.dropout(pooler_output)
        if self.Q_type == 'YN':
            preds_batch = self.classifier(x)
        else:
            preds = []
            for i in range(15):
                preds.append(self.classifier[i](x))
            
            preds_batch = []
            for i in range(x.shape[0]):
                pred_per_instance = []
                for pred in preds:
                    pred_per_instance.append(pred[i])
                pred_per_instance = torch.vstack(pred_per_instance)
                preds_batch.append(pred_per_instance)
            preds_batch = torch.stack(preds_batch)
        return preds_batch




class SpartunFRLoss(nn.Module):
    def __init__(self):
        super(SpartunFRLoss, self).__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, preds, labels):
        loss = 0
        for i in range(preds.shape[0]):
            loss += self.loss_fn(preds[i], labels[i])
        return loss
        
        
    

class GAT(nn.Module):
    def __init__(self, embedding_size, num_layers=3):
        super(GAT, self).__init__()
        self.conv = nn.ModuleList([GATConv(embedding_size, embedding_size, add_self_loops=False, edge_dim=embedding_size) \
                                   for i in range(num_layers)])

    def forward(self, x, edge_index, edge_features, batch):
        # 1. Obtain node embeddings 
        for i,ConvGnn in enumerate(self.conv):
            x = ConvGnn(x, edge_index, edge_features)
            if i<len(self.conv)-1:
                x = x.relu()
        # x = self.conv1(x, edge_index, edge_features)
        # x = x.relu()
        # x = self.conv2(x, edge_index, edge_features)
        # x = x.relu()
        # x = self.conv3(x, edge_index, edge_features)
        
        return x
    
    
    

class GCN(nn.Module):
    def __init__(self, embedding_size, num_layers=3):
        super(GCN, self).__init__()
        # self.conv1 = GCNConv(embedding_size, embedding_size)
        # self.conv2 = GCNConv(embedding_size, embedding_size)
        # self.conv3 = GCNConv(embedding_size, embedding_size)
        self.conv = nn.ModuleList([GCNConv(embedding_size, embedding_size, add_self_loops=False) \
                                   for i in range(num_layers)])

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        for i,ConvGnn in enumerate(self.conv):
            x = ConvGnn(x, edge_index)
            if i<len(self.conv)-1:
                x = x.relu()
        # x = self.conv1(x, edge_index)
        # x = x.relu()
        # x = self.conv2(x, edge_index)
        # x = x.relu()
        # x = self.conv3(x, edge_index)
        
        return x




class GraphC(nn.Module):
    def __init__(self, embedding_size, num_layers):
        super(GraphC, self).__init__()
        # self.conv1 = GraphConv(embedding_size, embedding_size)
        # self.conv2 = GraphConv(embedding_size, embedding_size)
        # self.conv3 = GraphConv(embedding_size, embedding_size)
        self.conv = nn.ModuleList([GraphConv(embedding_size, embedding_size, add_self_loops=False) \
                                   for i in range(num_layers)])

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        for i,ConvGnn in enumerate(self.conv):
            x = ConvGnn(x, edge_index)
            if i<len(self.conv)-1:
                x = x.relu()      
        # x = self.conv1(x, edge_index)
        # x = x.relu()
        # x = self.conv2(x, edge_index)
        # x = x.relu()
        # x = self.conv3(x, edge_index)
        
        return x



class GCNII(nn.Module):
    def __init__(self, embedding_size, num_layers):
        super(GCNII, self).__init__()
        # self.conv1 = GraphConv(embedding_size, embedding_size)
        # self.conv2 = GraphConv(embedding_size, embedding_size)
        # self.conv3 = GraphConv(embedding_size, embedding_size)
        self.conv = nn.ModuleList([GCN2Conv(embedding_size, alpha=0.5, add_self_loops=False) \
                                   for i in range(num_layers)])

    def forward(self, x, x_0, edge_index, batch):
        # 1. Obtain node embeddings
        for i,ConvGnn in enumerate(self.conv):
            x = ConvGnn(x, x_0, edge_index)
            if i<len(self.conv)-1:
                x = x.relu()      
        # x = self.conv1(x, edge_index)
        # x = x.relu()
        # x = self.conv2(x, edge_index)
        # x = x.relu()
        # x = self.conv3(x, edge_index)
        
        return x    
    
    
    


        
        


class Overall_model_stepgame(nn.Module):
    def __init__(self, device, model_name, aggregator, GNN_comparison=False, GNN_type=None, num_layers=3, seed_val=42,
                 embedding_size=768, proj_size_filler=300, 
                 proj_size_role=300, num_classes=9):
        super(Overall_model_stepgame, self).__init__()
        seed_torch(seed_val)
        self.PLM = AutoModel.from_pretrained(model_name)
        self.Linear_projection = nn.Linear(embedding_size, embedding_size)
        self.device = device
        self.reasoner = My_reasoner(device, input_features=embedding_size, proj_size_filler=embedding_size, 
                                    proj_size_role=embedding_size, aggregator=aggregator)
        self.classifier = nn.ModuleList([nn.Linear(embedding_size, embedding_size//2),
                                         nn.Linear(embedding_size//2, embedding_size//2),
                                         nn.Linear(embedding_size//2, num_classes)])
        self.layer_norm = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(p=0.2)
        
        self.GNN_comparison = GNN_comparison
        if GNN_comparison:
            self.GNN_type = GNN_type
            
            if GNN_type == "GCN":
                self.GNN = GCN(embedding_size=embedding_size, num_layers=num_layers)
            elif GNN_type == 'GAT':
                self.GNN = GAT(embedding_size=embedding_size, num_layers=num_layers)
            elif GNN_type == 'GraphC':
                self.GNN = GraphC(embedding_size=embedding_size, num_layers=num_layers)
            elif GNN_type == 'GCNII':
                self.GNN = GCNII(embedding_size=embedding_size, num_layers=num_layers)
        
    
    
    
    def forward(self, input_ids, attention_mask, token_type_ids, 
                S_entities_mask, edge_index, segment_ids, Q_entities_idx):
        outputs = self.PLM(input_ids, attention_mask, token_type_ids)
        pooler_output = outputs.pooler_output
        last_hidden_states = outputs.last_hidden_state
        last_hidden_states_PLM = last_hidden_states.clone()
        last_hidden_states = self.Linear_projection(last_hidden_states)

        node_features_set, edge_features_set = [], []
        for i in range(last_hidden_states.shape[0]):
            node_features, edge_features = [], [] 
            idx=0
            while(1):
                boolean_idx = torch.eq(S_entities_mask[i,:], idx)
                if boolean_idx.sum()==0:
                    break
                x = last_hidden_states[i, boolean_idx, :]
                x = torch.mean(x, 0, keepdim=True)
                node_features.append(x)
                idx+=1
            
            # idx=0
            # while(1):
            #     boolean_idx = torch.eq(edge_attr[i,:],idx)
            #     if boolean_idx.sum()==0:
            #         break
            #     x = last_hidden_states[i, boolean_idx, :]
            #     x = torch.mean(x, 0, keepdim=True)
            #     edge_features.extend([x]*2)
            #     edge_features.extend([torch.zeros(x.shape).to(self.device)]*2)
            #     idx+=1
            
            node_features_set.append(torch.cat(node_features,0))
            # edge_features_set.append(torch.cat(edge_features,0))
            
            edge_idx = edge_index[i]
            for j in range(edge_idx.shape[1]):
                if edge_idx[0,j]!=-100:
                    edge_features.append(pooler_output[i] if edge_idx[0,j]!=edge_idx[1,j] \
                                         else torch.zeros(pooler_output[i].shape).to(self.device))
            edge_features_set.append(torch.vstack(edge_features))
            

        # print(node_features_set, len(node_features_set))
        # print(edge_features_set , len(edge_features_set))
        
        if not self.GNN_comparison:
            target_node_set = []
            for i,node_features in enumerate(node_features_set):
                target_node_set.append(node_features[Q_entities_idx[i,0].item()])
            node_features_set_new = self.reasoner(node_features_set, edge_index, edge_features_set, target_node_set)
        else:
            data_list = []
            for node_features,edge_idx,edge_features in zip(node_features_set, edge_index, edge_features_set):
                source = torch.masked_select(edge_idx[0], edge_idx[0].ge(0))
                destination = torch.masked_select(edge_idx[1], edge_idx[1].ge(0))
                if self.GNN_type == 'GAT':
                    data = Data(x=node_features, edge_index=torch.vstack([source, destination]), edge_attr=edge_features)
                else:
                    data = Data(x=node_features, edge_index=torch.vstack([source, destination]))
                data_list.append(data)
                
            loader = DataLoader(data_list, batch_size=32, shuffle=False)
            for data in loader:
                if self.GNN_type == 'GAT':
                    node_features_new = self.GNN(data.x, data.edge_index, data.edge_attr, data.batch)
                elif self.GNN_type == 'GCNII':
                    node_features_new = self.GNN(data.x, data.x, data.edge_index, data.batch)
                else:
                    node_features_new = self.GNN(data.x, data.edge_index, data.batch)
                    
            node_features_set_new = []
            for i in range(data.batch[-1]+1):
                node_features_set_new.append(node_features_new[data.batch==i])
            
        # print(node_features_set_new, node_features_set_new.shape)
        
        # node_features_set_new = node_features_set
        attn_outputs = []
        for i,node_features_new in enumerate(node_features_set_new):
            idx=0
            while(1):
                boolean_idx = torch.eq(S_entities_mask[i,:], idx)
                if boolean_idx.sum()==0:
                    break
                row_num = last_hidden_states_PLM[i,boolean_idx,:].shape[0]
                last_hidden_states_PLM[i,boolean_idx,:] += torch.vstack([node_features_new[idx]]*row_num)
                # last_hidden_states_PLM[i,boolean_idx,:] += torch.vstack([torch.randn(node_features_new[idx].shape).to(self.device)]*row_num)
                idx+=1
            query = last_hidden_states_PLM[i,torch.eq(segment_ids[i],2),:]
            key = last_hidden_states_PLM[i,torch.eq(segment_ids[i],1),:]
            value = last_hidden_states_PLM[i,torch.eq(segment_ids[i],1),:]
            x = torch.div(torch.matmul(query,key.T), key.shape[1]**(0.5))
            attn_output = torch.matmul(x.softmax(dim=-1), value)
            attn_output = torch.sum(attn_output, dim=0)
            attn_outputs.append(attn_output)
            
            
        x = torch.stack(attn_outputs)
        # print(attn_outputs, attn_outputs.shape)
        x = self.layer_norm(x)
        x = self.classifier[0](x)
        x.relu()
        x = self.dropout(x)
        x = self.classifier[1](x)
        x.relu()
        x = self.dropout(x)
        x = self.classifier[2](x)
        
        return x
        
        





class Overall_model_spartun(nn.Module):
    def __init__(self, device, model_name, aggregator, Q_type, 
                 seed_val=42, embedding_size=768, proj_size_filler=300, 
                 proj_size_role=300, num_classes=2):
        super(Overall_model_spartun, self).__init__()
        seed_torch(seed_val)
        self.PLM = AutoModel.from_pretrained(model_name)
        self.Linear_projection = nn.Linear(embedding_size, embedding_size)
        self.device = device
        self.Q_type = Q_type
        self.reasoner = My_reasoner(device, input_features=embedding_size, proj_size_filler=embedding_size, 
                                    proj_size_role=embedding_size, aggregator=aggregator)
        
        if Q_type == 'YN': 
            self.classifier = nn.ModuleList([nn.Linear(embedding_size, embedding_size//2),
                                             nn.Linear(embedding_size//2, embedding_size//2),
                                             nn.Linear(embedding_size//2, num_classes)])
        else:
            self.classifier = nn.ModuleList(nn.Linear(embedding_size, num_classes) for i in range(15))
            
        self.target_node_extractor = nn.Linear(embedding_size, embedding_size)
        self.layer_norm = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(p=0.2)
        
    
    
    def forward(self, input_ids, attention_mask, token_type_ids, 
                S_entities_mask, edge_index, segment_ids):
        outputs = self.PLM(input_ids, attention_mask, token_type_ids)
        pooler_output = outputs.pooler_output
        last_hidden_states = outputs.last_hidden_state
        last_hidden_states_PLM = last_hidden_states.clone()
        last_hidden_states = self.Linear_projection(last_hidden_states)

        node_features_set, edge_features_set = [], []
        for i in range(last_hidden_states.shape[0]):
            node_features, edge_features = [], [] 
            idx=0
            
            while(1):
                boolean_idx = torch.eq(S_entities_mask[i,:], idx)
                if boolean_idx.sum()==0:
                    break
                x = last_hidden_states[i, boolean_idx, :]
                x = torch.mean(x, 0, keepdim=True)
                node_features.append(x)
                idx+=1
            
            node_features_set.append(torch.cat(node_features,0))
            # print(S_entities_mask[i,:], S_entities_mask[i,:].shape)
            # print(node_features_set[i], node_features_set[i].shape)
            # print(edge_index[i], edge_index[i].shape)
            
            edge_idx = edge_index[i]
            for j in range(edge_idx.shape[1]):
                if edge_idx[0,j]!=-100:
                    edge_features.append(pooler_output[i] if edge_idx[0,j]!=edge_idx[1,j] \
                                         else torch.zeros(pooler_output[i].shape).to(self.device))
            edge_features_set.append(torch.vstack(edge_features))
            
        # print(node_features_set, len(node_features_set))
        # print(edge_features_set , len(edge_features_set))
        
        target_node_set = []
        for i in range(last_hidden_states.shape[0]):
            x = torch.sum(last_hidden_states_PLM[i,torch.eq(segment_ids[i],2),:], dim=0)
            x = self.target_node_extractor(x)
            target_node_set.append(x)
        node_features_set_new = self.reasoner(node_features_set, edge_index, edge_features_set, target_node_set)
    
        attn_outputs = []
        for i,node_features_new in enumerate(node_features_set_new):
            idx=0
            while(1):
                boolean_idx = torch.eq(S_entities_mask[i,:], idx)
                if boolean_idx.sum()==0:
                    break
                row_num = last_hidden_states_PLM[i,boolean_idx,:].shape[0]
                last_hidden_states_PLM[i,boolean_idx,:] += torch.vstack([node_features_new[idx]]*row_num)
                idx+=1
            query = last_hidden_states_PLM[i,torch.eq(segment_ids[i],2),:]
            key = last_hidden_states_PLM[i,torch.eq(segment_ids[i],1),:]
            value = last_hidden_states_PLM[i,torch.eq(segment_ids[i],1),:]
            x = torch.div(torch.matmul(query,key.T), key.shape[1]**(0.5))
            attn_output = torch.matmul(x.softmax(dim=1), value)
            attn_output = torch.sum(attn_output, dim=0)
            attn_outputs.append(attn_output)
            
            
        x = torch.stack(attn_outputs)
        # print(attn_outputs, attn_outputs.shape)
        x = self.layer_norm(x)
        if self.Q_type == 'YN':
            x = self.classifier[0](x)
            x.relu()
            x = self.dropout(x)
            x = self.classifier[1](x)
            x.relu()
            x = self.dropout(x)
            preds_batch = self.classifier[2](x)
        else:
            preds = []
            for i in range(15):
                preds.append(self.classifier[i](x))
            
            preds_batch = []
            for i in range(x.shape[0]):
                pred_per_instance = []
                for pred in preds:
                    pred_per_instance.append(pred[i])
                pred_per_instance = torch.vstack(pred_per_instance)
                preds_batch.append(pred_per_instance)
            preds_batch = torch.stack(preds_batch)
        return preds_batch




    

    


class My_reasoner(nn.Module):
    
    def __init__(self, device, input_features, aggregator,
                 proj_size_filler, proj_size_role, num_classes=9):
        super(My_reasoner, self).__init__()
        
        
        self.MLP_filler = nn.ModuleList([nn.Linear(input_features*3, input_features),
                                          nn.Linear(input_features, input_features),
                                          nn.Linear(input_features, proj_size_filler)])
        
        self.filler_aggregation = nn.LSTM(input_size=input_features, hidden_size=1024, num_layers=1, proj_size=input_features)
        self.filler_projection = nn.ModuleList([nn.Linear(proj_size_filler, proj_size_filler),
                                                nn.Linear(proj_size_filler, proj_size_filler),
                                                nn.Linear(proj_size_filler, proj_size_filler)])

        self.linears_inference = nn.ModuleList([nn.Linear(input_features*2+proj_size_filler, input_features), 
                                                nn.Linear(input_features, input_features),
                                                nn.Linear(input_features, input_features)])
        
        self.dropout = nn.Dropout(p=0.2)
        self.layer_norm = nn.LayerNorm(proj_size_filler)
        self.device = device
        self.aggregator = aggregator
        self.proj_size_filler = proj_size_filler
        self.proj_size_role = proj_size_role
        



    def initialization(self, node_features_set, edge_features_set, edge_index):  
        batch_MLP_input = []
        for node_features,edge_features,edge_idx in zip(node_features_set,edge_features_set,edge_index):
            input_seq = []
            for j in range(edge_idx.shape[1]):
                if edge_idx[0,j]!=-100:
                    input_seq.append(torch.hstack((node_features[edge_idx[0,j].item()],
                                                    edge_features[j],
                                                    node_features[edge_idx[1,j].item()])))
                        
                
            input_seq = torch.vstack(input_seq)
            batch_MLP_input.append(input_seq)
        
        batch_MLP_input = torch.nn.utils.rnn.pad_sequence(batch_MLP_input, batch_first=True)
        x = self.MLP_filler[0](batch_MLP_input)
        x.relu()
        x = self.dropout(x)
        x = self.MLP_filler[1](x)
        x.relu()
        x = self.dropout(x)
        x = self.MLP_filler[2](x)
        
        role_filler_memory_set = []
        for i,(node_features,edge_idx) in enumerate(zip(node_features_set,edge_index)):
            role_filler_memory = zeros(node_features.shape[0], 
                                       self.proj_size_filler, 
                                       self.proj_size_role).to(self.device)
            for j in range(edge_idx.shape[1]):
                if edge_idx[0,j]!=-100:
                    filler = x[i,j,:]
                    # filler = torch.randn(self.proj_size_filler).to(self.device)
                    role_filler_memory[edge_idx[0,j].item()] += torch.outer(filler, node_features[edge_idx[1,j].item()])
            role_filler_memory_set.append(role_filler_memory)
        
        return role_filler_memory_set
        
    
    
    
    def collect_long_dependancy(self, role_filler_memory, node_features, path):
        # forward and backward
        original_role_filler_memory = role_filler_memory.clone()               
        if len(path)>1:
            # print(path)
            for r in range(2):
                filler_set = []
                for idx,i in enumerate(path):
                    if idx<len(path)-1:
                        filler_set.append(torch.matmul(original_role_filler_memory[i,:,:], 
                                                        node_features[path[idx+1],:]))
                filler_set = torch.vstack(filler_set)
                
                if self.aggregator == 'summation':
                    filler_aggregated = torch.sum(filler_set,dim=0)
                elif self.aggregator == 'mean':
                    filler_aggregated = torch.mean(filler_set,dim=0)
                elif self.aggregator == 'max_pooling':
                    filler_aggregated = torch.max(filler_set,dim=0)[0]
                elif self.aggregator == 'LSTM':
                    _, (hn, _) = self.filler_aggregation(filler_set)
                    filler_aggregated = hn.squeeze()
                
                # print(filler_aggregated)
                x = self.filler_projection[0](filler_aggregated)
                x.relu()
                x = self.dropout(x)
                x = self.filler_projection[1](x)
                x.relu()
                x = self.dropout(x)
                x = self.filler_projection[2](x)
                x = self.layer_norm(x + filler_aggregated)

                role_filler_memory[path[0],:,:] += torch.outer(x, node_features[path[-1],:])
                path.reverse()            
                
        return role_filler_memory
    
    
    
    
    def shortest_path(self, adj_list, node_idx1, node_idx2):

        path_list = [[node_idx1]]
        path_index = 0
        # To keep track of previously visited nodes
        previous_nodes = {node_idx1}
        if node_idx1 == node_idx2:
            current_path = path_list[0]
        else:
            while path_index < len(path_list):
                current_path = path_list[path_index]
                last_node = current_path[-1]
                # Search goal node
                
                if node_idx2 in adj_list[last_node]:
                    current_path.append(node_idx2)
                    break
                # Add new paths
                for next_node in adj_list[last_node]:
                    if not next_node in previous_nodes:
                        new_path = current_path[:]
                        new_path.append(next_node)
                        path_list.append(new_path)
                        # To avoid backtracking
                        previous_nodes.add(next_node)                
                path_index += 1
            # No path is found
            if path_index >= len(path_list):
                current_path = []
            
        return current_path
    
    
    
    
    def propogation(self, role_filler_memory_set, adj_list_set, node_features_set):
        original_role_filler_memory_set = copy.copy(role_filler_memory_set)
        for idx,(adj_list,node_features) in enumerate(zip(adj_list_set,node_features_set)):
            nodes_set = set(adj_list.keys())
            path_covered = []
            for node,neighbors in adj_list.items():
                non_neighbors = nodes_set.difference(set(neighbors))
                for non_neighbor_node in non_neighbors:
                    if [node, non_neighbor_node] not in path_covered or [non_neighbor_node, node] not in path_covered:
                        path = self.shortest_path(adj_list, node, non_neighbor_node)
                        role_filler_memory_set[idx] = self.collect_long_dependancy(original_role_filler_memory_set[idx], 
                                                                                   node_features, 
                                                                                   path)
                        
                        path_covered.append([node, non_neighbor_node])
                        path_covered.append([non_neighbor_node, node])
            
        return role_filler_memory_set
    
    
    
        
    def inference(self, role_filler_memory_set, node_features_set, target_node_set):
        node_features_set_new  = []
        for role_filler_memory,node_features,target_node in zip(role_filler_memory_set,node_features_set,target_node_set):
            # print(role_filler_memory,node_features,Q_entities)
            retrieved_fillers = torch.matmul(role_filler_memory, target_node)
            # retrieved_fillers = torch.matmul(role_filler_memory, torch.randn(self.proj_size_filler).to(self.device))
            retrieved_fillers = self.layer_norm(retrieved_fillers)
            target_node_features = torch.vstack([target_node]*node_features.shape[0])
            input_seq = torch.hstack((node_features, retrieved_fillers, target_node_features))
            x = self.linears_inference[0](input_seq)
            x.relu()
            x = self.dropout(x)
            x = self.linears_inference[1](x)
            x.relu()
            x = self.dropout(x)
            x = self.linears_inference[2](x)
            node_features_set_new.append(x)
        
        return node_features_set_new
        



    def forward(self, node_features_set, edge_index, edge_features_set, target_node_set):            
        
        role_filler_memory_set = self.initialization(node_features_set, edge_features_set, edge_index)   
        adj_list_set = []
        for i in range(edge_index.shape[0]):
            adj_list = {}
            for j in range(edge_index[i].shape[1]):
                adj_list[int(edge_index[i,0,j].item())] = []
            for j in range(edge_index[i].shape[1]):
                node1 = int(edge_index[i,0,j].item())
                node2 = int(edge_index[i,1,j].item())
                if node2 not in adj_list[node1]:
                    adj_list[node1].append(node2)
            adj_list_set.append(adj_list)
                
        role_filler_memory_set = self.propogation(role_filler_memory_set, adj_list_set, node_features_set)
        node_features_set_new = self.inference(role_filler_memory_set, node_features_set, target_node_set)
        
        return node_features_set_new