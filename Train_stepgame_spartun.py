

import pandas as pd
import torch
import pickle
import wandb
import json
import os
from Models import Overall_model_stepgame, Overall_model_spartun, SpartunFRLoss
from Preprocessing import get_tensor_dataset_stepgame, get_tensor_dataset_spartun
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import random




def train_loop(dataloader, model, loss_fn, optimizer, device, wandb=None):
    size = len(dataloader.dataset)
    model.to(device)
    model.train()
    for batch, data in enumerate(dataloader):
        if len(data)==8:
            input_ids, attention_mask, token_type_ids, \
            S_entities_mask, edge_index, edge_attr, \
            Q_entities_idx, label = data
            
            pred = model(input_ids.to(device), attention_mask.to(device), token_type_ids.to(device), 
                         S_entities_mask.to(device), edge_index.to(device), edge_attr.to(device), 
                         Q_entities_idx.to(device))
        else:
            input_ids, attention_mask, token_type_ids, \
            S_entities_mask, edge_index, edge_attr, \
            label = data

            pred = model(input_ids.to(device), attention_mask.to(device), token_type_ids.to(device), 
                         S_entities_mask.to(device), edge_index.to(device), edge_attr.to(device))
        
        # print(pred, label)
        label = label.type(torch.LongTensor) 
        loss = loss_fn(pred, label.to(device))
    
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss, current = loss.item(), (batch + 1) * input_ids.shape[0]
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")




def postprocessing(preds):
    candidate_answers = ['left', 'right', 'above', 'below', 'behind', 'front', 
                         'near', 'far', 'dc', 'ec', 'po', 'tpp', 'ntpp', 'tppi', 
                         'ntppi']
    
    for i in range(preds.shape[0]):
        for j in range(0,preds.shape[-1],2):
            if j>=10:
                if preds[i,j+1]==1 and preds[i,j+2]==1:
                    preds[i,j+2]=0    
                if j+2==preds.shape[-1]-1:
                    break
            else:
                if preds[i,j]==1 and preds[i,j+1]==1:
                    preds[i,j+1]=0
    
    return preds





def test_loop(dataloader, model, loss_fn, device, oper='validation', wandb=None):
    num_batches = len(dataloader)
    # F1 = evaluate.load("f1")

    model.to(device)
    model.eval()
    final_results = {'accuracy':0, f'{oper}_loss':0}
    with torch.no_grad():
        for data in dataloader:
            if len(data)==8:
                input_ids, attention_mask, token_type_ids, \
                S_entities_mask, edge_index, edge_attr, \
                Q_entities_idx, label = data
                
                pred = model(input_ids.to(device), attention_mask.to(device), token_type_ids.to(device), 
                S_entities_mask.to(device), edge_index.to(device), edge_attr.to(device), 
                Q_entities_idx.to(device))
            else:
                input_ids, attention_mask, token_type_ids, \
                S_entities_mask, edge_index, edge_attr, \
                label = data
    
                pred = model(input_ids.to(device), attention_mask.to(device), token_type_ids.to(device), 
                             S_entities_mask.to(device), edge_index.to(device), edge_attr.to(device))
            
            label = label.type(torch.LongTensor)
            predictions = torch.argmax(pred.softmax(dim=-1), dim=-1)
            references = label.to(device)
            # print(predictions, references)
            if references.dim()==1:
                accuracy = torch.eq(predictions, references).sum()/references.shape[0]
            else:
                predictions = postprocessing(predictions)
                accuracy = 0
                for i in range(references.shape[0]):
                    accuracy += torch.eq(predictions[i], references[i]).sum()/references.shape[-1]
                accuracy /= references.shape[0]
            # f1 = F1.compute(references=references, predictions=predictions, average='micro')
            final_results['accuracy'] += accuracy.item()
            # final_results['f1'] += f1['f1']
            final_results[f'{oper}_loss'] += loss_fn(pred, label.to(device)).item()
            
    for key in final_results.keys():        
        final_results[key] /= num_batches
    print(final_results)
    
    return final_results




if __name__=="__main__":
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # dataset_used = 'StepGame-main'
    dataset_used = 'SPARTUN'
    
    if dataset_used == 'StepGame-main':
        lr = 5e-5
        PLM_lr = 6e-6
        batch_size = 8
    else:
        lr = 1e-4
        PLM_lr = 8e-6
        batch_size = 16
    epochs = 10
    scheduler_patience = 1
    delta = 1e-3
    factor = 0.1
    early_stopping_patience = 3
    ablation_study = 'retrieval'
    GNN_type = 'GraphC'
    GNN_comparison = True
    num_layers = 7
    # model_name = 'bert-base-cased'
    # model_name = 'roberta-base'
    model_name = 'albert-base-v2'
    filler_aggregator = 'LSTM' #[summation, mean, max_pooling, LSTM]
    if dataset_used == 'SPARTUN':
        Q_type = 'YN'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
   
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    
    if dataset_used == 'StepGame-main':
        with open('labels_set.pkl', 'rb') as f:
            labels_set = pickle.load(f)
        
        dftrain = pd.read_json(f'{dataset_used}/Dataset/TrainVersion/train.json')
        dftrain = dftrain.swapaxes(0,1)
        dftrain = dftrain.dropna()
        train_dataset = get_tensor_dataset_stepgame(dftrain, tokenizer, labels_set, model_name)
        
        eval_dataset = []
        for i in range(1,6):
            eval_name = f'qa{i}_valid.json'
            dfeval = pd.read_json(f'{dataset_used}/Dataset/TrainVersion/{eval_name}')
            dfeval = dfeval.swapaxes(0,1)
            dfeval = dfeval.dropna()
            eval_dataset.extend(get_tensor_dataset_stepgame(dfeval, tokenizer, labels_set, model_name))
        
        test_datasets = []
        for i in range(1,11):
            test_name = f'qa{i}_test.json'
            dftest = pd.read_json(f'{dataset_used}/Dataset/TrainVersion/{test_name}')
            dftest = dftest.swapaxes(0,1)
            dftest = dftest.dropna()
            test_datasets.append(get_tensor_dataset_stepgame(dftest, tokenizer, labels_set, model_name))
            
    else:
        with open(f'{dataset_used}/train.json','r') as f:
            dataset = json.load(f)
        train_dataset = get_tensor_dataset_spartun(dataset, tokenizer, model_name=model_name, question_type=Q_type)
        
        with open(f'{dataset_used}/dev.json','r') as f:
            dataset = json.load(f)
        eval_dataset = get_tensor_dataset_spartun(dataset, tokenizer, model_name=model_name, question_type=Q_type)
        
        with open(f'{dataset_used}/test.json','r') as f:
            dataset = json.load(f)
        test_dataset = get_tensor_dataset_spartun(dataset, tokenizer, model_name=model_name, question_type=Q_type)
    
    
    random.seed(42)
    random.shuffle(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size) 
    

    folder = f'saved_model/{dataset_used}/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    if dataset_used == 'StepGame-main' :
        model = Overall_model_stepgame(device=device, model_name=model_name, aggregator=filler_aggregator, 
                                       GNN_comparison=GNN_comparison, GNN_type=GNN_type, num_layers=num_layers)
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        model = Overall_model_spartun(device=device, model_name=model_name, aggregator=filler_aggregator, Q_type=Q_type)
        if Q_type == 'FR':
            loss_fn = SpartunFRLoss()
        else:
            loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([{'params': model.PLM.parameters(), 'lr': PLM_lr},
                                  {'params': model.Linear_projection.parameters()},
                                  {'params': model.reasoner.parameters()},
                                  {'params': model.classifier.parameters()}], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, 
                                                           patience=scheduler_patience,
                                                           threshold=delta if delta else 0)
    
    best_loss = 100
    cnt=0
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, device)
        results = test_loop(eval_dataloader, model, loss_fn, device)
        loss = results['validation_loss']
        scheduler.step(loss)
        
        if loss<best_loss-delta:
            best_loss = loss
            cnt=0
        else:
            cnt+=1
        
        if cnt>=early_stopping_patience:
            print(f'Early stopping counter:{cnt} out of {early_stopping_patience}')
            print('Early stopping condition fulfilled')
            break
        
            
    
    
    columns, data = [], []
    if dataset_used == 'StepGame-main':
        for i,test_dataset in enumerate(test_datasets):
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
            results = test_loop(test_dataloader, model, loss_fn, device, oper='test')
            columns.append(f'qa{i+1}_test')
            data.append(results['accuracy'])
    else:
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
        results = test_loop(test_dataloader, model, loss_fn, device, oper='test')
        columns.append(Q_type)
        data.append(results['accuracy'])
    
    print(columns)
    print(data)
        
    torch.save(model.state_dict(), f'{folder}/model_weights.pth')
    print("Done!")
    
