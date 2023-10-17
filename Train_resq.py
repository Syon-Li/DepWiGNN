


import os
import json
import random
import torch
import wandb
from Preprocessing import get_tensor_dataset_baseline_spartun_resq, get_tensor_dataset_resq
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from Models import Baseline_model_spartun, Overall_model_spartun
from Train_stepgame_spartun import train_loop, test_loop
from Baselines_train_eval import train_loop as train_loop_baseline, test_loop as test_loop_baseline



if __name__=="__main__":
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    dataset_used = 'ReSQ'
    
    lr = 4e-5
    PLM_lr = 5e-5
    batch_size = 16
    epochs = 50
    scheduler_patience = 1
    delta = 1e-3
    factor = 0.1
    early_stopping_patience = 3
    baseline = False
    use_pretrain_spartun = False
    model_name = 'bert-base-cased'
    # model_name = 'roberta-base'
    # model_name = 'albert-base-v2'
    filler_aggregator = 'LSTM'
    Q_type = 'YN'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
   
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    with open(f'{dataset_used}/train_resq.json','r') as f:
        train_dataset = json.load(f)
    
    with open(f'{dataset_used}/dev_resq.json','r') as f:
        eval_dataset = json.load(f)
    
    with open(f'{dataset_used}/test_resq.json','r') as f:
        test_dataset = json.load(f)
    
    
    if baseline:
        train_dataset = get_tensor_dataset_baseline_spartun_resq(train_dataset, tokenizer, question_type=Q_type)
        eval_dataset = get_tensor_dataset_baseline_spartun_resq(eval_dataset, tokenizer, question_type=Q_type)
        test_dataset = get_tensor_dataset_baseline_spartun_resq(test_dataset, tokenizer, question_type=Q_type)
    else:
        train_dataset = get_tensor_dataset_resq(train_dataset, tokenizer, model_name=model_name)
        eval_dataset = get_tensor_dataset_resq(eval_dataset, tokenizer, model_name=model_name)
        test_dataset = get_tensor_dataset_resq(test_dataset, tokenizer, model_name=model_name)
    
    
    random.seed(42)
    random.shuffle(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size) 
    
    
    folder = f'saved_model/{dataset_used}'
    
    
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    
    if baseline:
        model = Baseline_model_spartun(device=device, model_name=model_name, Q_type=Q_type)
        model.load_state_dict(torch.load('saved_model/SPARTUN//model_weights.pth'))
        optimizer = torch.optim.Adam(model.parameters(), lr=PLM_lr)
    else:
        model = Overall_model_spartun(device=device, model_name=model_name, aggregator=filler_aggregator, Q_type=Q_type)
        if use_pretrain_spartun:
            model.load_state_dict(torch.load('saved_model/SPARTUN/model_weights.pth'))
        optimizer = torch.optim.Adam([{'params': model.PLM.parameters(), 'lr': PLM_lr},
                                      {'params': model.Linear_projection.parameters()},
                                      {'params': model.reasoner.parameters()},
                                      {'params': model.classifier.parameters()}], lr=lr)
    
        
    loss_fn = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, 
                                                           patience=scheduler_patience,
                                                           threshold=delta if delta else 0)
    
    best_loss = 100
    cnt=0
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        if not baseline:
            train_loop(train_dataloader, model, loss_fn, optimizer, device)
            results = test_loop(eval_dataloader, model, loss_fn, device)
        else:
            train_loop_baseline(train_dataloader, model, loss_fn, optimizer, device)
            results = test_loop_baseline(eval_dataloader, model, loss_fn, device)
        
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
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    if not baseline:
        results = test_loop(test_dataloader, model, loss_fn, device, oper='test', wandb=wandb)
    else:
        results = test_loop_baseline(test_dataloader, model, loss_fn, device, oper='test', wandb=wandb)
    columns.append(Q_type)
    data.append(results['accuracy'])
    
    print(columns)
    print(data)
        
    torch.save(model.state_dict(), f'{folder}/model_weights.pth')
    print("Done!")