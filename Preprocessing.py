

import re
from tqdm.auto import tqdm
from typing import List
import nltk
import json
import torch
from transformers import AutoTokenizer

    



def get_tensor_dataset_stepgame(df,tokenizer,labels_set,model_name):
    tb = [chr(i) for i in range(ord('A'), ord('Z')+1)]
    padding_id = -100
    max_length = 400
    dataset = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Tokenizing data'):
        S, Q, y = row['story'], row['question'], row['label']            
        
        encoded_dict = tokenizer(" ".join(S),
                                 Q,
                                 return_token_type_ids=True,
                                 max_length=max_length,
                                 padding='max_length',
                                 return_tensors='pt')
        label = labels_set.index(y)
            
            
        S_entities = []
        S_entities_mask = torch.ones(encoded_dict['input_ids'].squeeze().shape)*padding_id
        
        # print(tokenizer.convert_ids_to_tokens(encoded_dict['input_ids'].squeeze()))
        # print(encoded_dict['input_ids'].squeeze())
        i = 0
        for idx,sent in enumerate(S):
            for token in tokenize(sent):
                if token in tb and token not in S_entities:
                    S_entities.append(token)
                    if 'bert-base-cased' == model_name:
                        token_ids = tokenizer.encode(token, add_special_tokens=False)
                        S_entities_mask[encoded_dict['input_ids'].squeeze()==token_ids[0]]=i
                    elif 'roberta-base' == model_name:
                        token_ids = tokenizer.encode(" ".join([token]*2), add_special_tokens=False)
                        for token_id in token_ids:
                            S_entities_mask[encoded_dict['input_ids'].squeeze()==token_id]=i
                    else:
                        token_ids = tokenizer.encode(token, add_special_tokens=False)
                        S_entities_mask[encoded_dict['input_ids'].squeeze()==token_ids[-1]]=i
                    # print(token, token_ids)
                    i+=1
        # print(S_entities_mask)
                
        segment_ids = [padding_id]
        segment_ids.extend([1]*len(tokenizer.encode(" ".join(S), add_special_tokens=False)))
        segment_ids.extend([padding_id]*(tokenizer.num_special_tokens_to_add(pair=True)//2))
        segment_ids.extend([2]*len(tokenizer.encode(Q, add_special_tokens=False)))
        #pad edge_attr
        if len(segment_ids)<max_length:
            segment_ids.extend([padding_id]*(max_length-len(segment_ids)))
        # print(edge_attr)
        
        
        entities = []
        for sent in S:
            entities.append(list(set([token for token in tokenize(sent) if token in tb])))
        edge_index, node1, node2 = [], [], []
        for s1 in entities:
            if len(s1)>1:
                node1.append(S_entities.index(s1[0]))
                node2.append(S_entities.index(s1[1]))
                node1.append(S_entities.index(s1[1]))
                node2.append(S_entities.index(s1[0]))
                node1.append(S_entities.index(s1[1]))
                node2.append(S_entities.index(s1[1]))
            node1.append(S_entities.index(s1[0]))
            node2.append(S_entities.index(s1[0]))
        
        #pad edge_index
        node1.extend([padding_id]*(200-len(node1)))
        node2.extend([padding_id]*(200-len(node2)))
        edge_index.append(node1)
        edge_index.append(node2)
        
                
        Q_entities_idx = []            
        for token in tokenize(Q):
            if token in tb:
                try:
                    Q_entities_idx.append(S_entities.index(token))
                except:
                    Q_entities_idx.append(0)
        
        dataset.append((encoded_dict['input_ids'].squeeze(),
                        encoded_dict['attention_mask'].squeeze(),
                        encoded_dict['token_type_ids'].squeeze(),
                        S_entities_mask,
                        torch.tensor(edge_index),
                        torch.tensor(segment_ids),
                        torch.tensor(Q_entities_idx),
                        label))
    # print(dataset) 
        
    return dataset





def get_tensor_dataset_spartun(dataset,tokenizer,model_name,question_type):
    max_length = 400
    padding_id = -100
    tensor_dataset = []
    grammar = r""""NP:{<ADJ|NOUN>+<NOUN|NUM>+}"""#'medium yellow apple' 'another medium yellow apple'
    parser = nltk.chunk.regexp.RegexpParser(grammar)
    candi_grammar = r""""NP:{<ADJ|NOUN>+<VERB>+<NOUN|NUM>}"""
    candi_parser = nltk.chunk.regexp.RegexpParser(candi_grammar)
    stemmer =  nltk.stem.porter.PorterStemmer()
    cnt=0
    for i,instances in tqdm(enumerate(dataset['data']), total=len(dataset['data']), desc='Tokenizing data'):
        S = instances['story'][0].lower()
    
        #Get entities
        S_tokens = nltk.word_tokenize(S)
        S_tagged = nltk.pos_tag(S_tokens, tagset='universal')
        tree = parser.parse(S_tagged)
        NP_set,entities = [], []
        for t in tree.subtrees(lambda t: t.height() < tree.height()):
            words = []
            for word,pos in list(t):
                words.append(word)    
            NP = (' '.join(words).lower())
            if NP not in NP_set:
                NP_set.append(NP)
        
        NP_candi_set = []
        tree = candi_parser.parse(S_tagged)
        for t in tree.subtrees(lambda t: t.height() < tree.height()):
            words = []
            for word,pos in list(t):
                if pos == 'VERB':
                    if word=='called' or word=='named':
                        words.append(word)
                    else:
                        words.clear()
                        break
                else:
                    words.append(word)
            if len(word)!=0:
                NP = (' '.join(words).lower())
                if NP not in NP_candi_set:
                    NP_candi_set.append(NP)
            
        entities.append([NP_set[0]])
        for NP in NP_set:
            flag = True
            for entity in entities:
                if NP not in entity:
                    NP_element = set(stemmer.stem(w) for w in NP.split())
                    for N_Phrase in entity:
                        # print(N_Phrase)
                        N_Phrase_element = set(stemmer.stem(w) for w in N_Phrase.split())
                        if NP_element.issubset(N_Phrase_element) or N_Phrase_element.issubset(NP_element):
                            #'medium orange apple' 'medium orange apple number two'
                            entity.append(NP)
                            flag = False
                            break
                    if flag==False:
                        break
                else:
                    flag = False
                    break
            if flag:
                entities.append([NP])
        
        for NP in NP_candi_set:
            for entity in entities:
                if NP not in entity:
                    NP_element = set(stemmer.stem(w) for w in NP.split())
                    for N_Phrase in entity:
                        N_Phrase_element = set(stemmer.stem(w) for w in N_Phrase.split())
                        if N_Phrase_element.issubset(NP_element):
                            entity.append(NP)
                            flag = False
                            break          
        # print(entities)
            
            
            
        for Q in instances['questions']:
            # print(Q['question'])
            encoded_dict = tokenizer(S.lower(),
                                      Q['question'].lower(),
                                      return_token_type_ids=True,
                                      max_length=max_length,
                                      padding='max_length',
                                      return_tensors='pt')
            if Q['q_type']==question_type:
                if question_type=='YN':
                    label = Q['candidate_answers'].index(Q['answer'][0])
                elif question_type=='FR':
                    label = torch.zeros(len(Q['candidate_answers']))
                    for answer in Q['answer']:
                        label[Q['candidate_answers'].index(answer.lower())] = 1    
                
                
                S_entities_mask = torch.ones(encoded_dict['input_ids'].squeeze().shape)*padding_id
                for i,entity in enumerate(entities):
                    for NP in entity:
                        if 'bert-base-cased' == model_name:
                            # print(NP)
                            token_ids = tokenizer.encode(NP, add_special_tokens=False, return_tensors='pt')
                            # print(token_ids)
                            S_entities_mask = find_ids(encoded_dict['input_ids'].squeeze(), token_ids.squeeze(), S_entities_mask, i)
                        elif 'roberta-base' == model_name:
                            token_ids_v1 = tokenizer.encode(NP, add_special_tokens=False, return_tensors='pt')
                            token_ids_v2 = tokenizer.encode(" "+NP, add_special_tokens=False, return_tensors='pt')
                            S_entities_mask = find_ids(encoded_dict['input_ids'].squeeze(), token_ids_v1.squeeze(), S_entities_mask, i)
                            S_entities_mask = find_ids(encoded_dict['input_ids'].squeeze(), token_ids_v2.squeeze(), S_entities_mask, i)
                        else:
                            token_ids = tokenizer.encode(NP, add_special_tokens=False, return_tensors='pt')
                            S_entities_mask = find_ids(encoded_dict['input_ids'].squeeze(), token_ids.squeeze(), S_entities_mask, i)

                
                segment_ids = [padding_id]
                segment_ids.extend([1]*len(tokenizer.encode(S.lower(), add_special_tokens=False)))
                segment_ids.extend([padding_id]*(tokenizer.num_special_tokens_to_add(pair=True)//2))
                segment_ids.extend([2]*len(tokenizer.encode(Q['question'].lower(), add_special_tokens=False)))
                #pad token_type_ids
                if len(segment_ids)<max_length:
                    segment_ids.extend([padding_id]*(max_length-len(segment_ids)))
                
                
                edge_index, node1, node2 = [], [], []
                edge_tuples = []
                traversed_entities = []
                for sent in nltk.sent_tokenize(S.lower()):
                    sent_entities = []
                    
                    if len(traversed_entities)>0:
                        DET_phrase = re.findall(r'this [a-z]+', sent)
                        if len(DET_phrase)>0:
                            obj_type = DET_phrase[0].split()[1]
                            if obj_type == 'block':
                                for i in range(len(traversed_entities)-1, -1, -1):
                                    for NP in traversed_entities[i]:
                                        if obj_type in NP:
                                            sent_entities.append(entities.index(traversed_entities[i]))
                                            break
                            else:
                                sent_entities.append(entities.index(traversed_entities[-1]))
                                
                        
                    for i,entity in enumerate(entities):
                        for NP in entity:
                            if NP in sent:
                                sent_entities.append(i)
                                traversed_entities.append(entity)
                                break
                    
                    sent_entities = list(set(sent_entities))
                    if len(sent_entities)>1:
                        # print(sent_entities)
                        for idx in range(len(sent_entities)-1):
                            for idx1 in range(idx+1,len(sent_entities)):
                                if (sent_entities[idx], sent_entities[idx1]) not in edge_tuples:
                                    edge_tuples.append((sent_entities[idx], sent_entities[idx1]))
                                    edge_tuples.append((sent_entities[idx1], sent_entities[idx]))
                #add self-loops
                for i,entity in enumerate(entities):
                    node1.append(i)
                    node2.append(i)
                for edge_tuple in edge_tuples:
                    node1.append(edge_tuple[0])
                    node2.append(edge_tuple[1])
                #pad edge_index
                node1.extend([padding_id]*(200-len(node1)))
                node2.extend([padding_id]*(200-len(node2)))
                edge_index.append(node1)
                edge_index.append(node2)
                
                # idx=0
                # while(1):
                #     boolean_idx = torch.eq(S_entities_mask, idx)
                #     if boolean_idx.sum()==0:
                #         break
                #     idx+=1
                
                # if idx!=len(entities):
                #     print(S)
                #     print(entities)
                #     print(NP_set)
                #     print(S_entities_mask)
                #     print(torch.tensor(edge_index), torch.tensor(edge_index).shape)
                #     cnt+=1
                # print(S)
                # print(entities)
                # print(S_entities_mask)
                # print(torch.tensor(edge_index), torch.tensor(edge_index).shape)
                tensor_dataset.append((encoded_dict['input_ids'].squeeze(),
                                       encoded_dict['attention_mask'].squeeze(),
                                       encoded_dict['token_type_ids'].squeeze(),
                                       S_entities_mask,
                                       torch.tensor(edge_index),
                                       torch.tensor(segment_ids),
                                       label))
    
    # print(cnt)
        
    return tensor_dataset
                
                




def find_ids(input_ids, token_ids, S_entities_mask, entity_id):
    for idx_input in range(input_ids.shape[0]):
        idx_token = 0
        temp_idx = idx_input
        # print(input_ids)
        # print(token_ids)
        while(idx_token<token_ids.shape[0] and input_ids[temp_idx]==token_ids[idx_token]):
            # print(temp_idx)
            idx_token += 1
            temp_idx += 1
        if idx_token==token_ids.shape[0]:
            for r in range(idx_input,temp_idx):
                S_entities_mask[r] = entity_id
    
    return S_entities_mask






def get_tensor_dataset_resq(dataset,tokenizer,model_name):
    max_length = 200
    padding_id = -100
    tensor_dataset = []
    grammar = r""""NP:{<ADJ|NOUN>?<NOUN|NUM>+}"""  
    parser = nltk.chunk.regexp.RegexpParser(grammar)
    stemmer =  nltk.stem.porter.PorterStemmer()
    cnt=0
    for i,instances in tqdm(enumerate(dataset['data']), total=len(dataset['data']), desc='Tokenizing data'):
        S = instances['story'][0].lower()
    
        #Get entities
        S_tokens = nltk.word_tokenize(S)
        S_tagged = nltk.pos_tag(S_tokens, tagset='universal')
        tree = parser.parse(S_tagged)       
        NP_set = []         
        for t in tree.subtrees(lambda t: t.height() < tree.height()):
            words = []
            for word,pos in list(t):
                words.append(word)    
            NP = (' '.join(words).lower())
            if NP not in NP_set:
                NP_set.append(NP)
        
        entities = []
        entities.append([NP_set[0]])
        for NP in NP_set:
            flag = True
            for entity in entities:
                if NP not in entity:
                    NP_element = set(stemmer.stem(w) for w in NP.split())
                    for N_Phrase in entity:
                        # print(N_Phrase)
                        N_Phrase_element = set(stemmer.stem(w) for w in N_Phrase.split())
                        if NP_element.issubset(N_Phrase_element) or N_Phrase_element.issubset(NP_element):
                            #'medium orange apple' 'medium orange apple number two'
                            entity.append(NP)
                            flag = False
                            break
                    if flag==False:
                        break
                else:
                    flag = False
                    break
            if flag:
                entities.append([NP])
        
        
        for Q in instances['questions']:
            # print(Q['question'])
            encoded_dict = tokenizer(S.lower(),
                                      Q['question'].lower(),
                                      return_token_type_ids=True,
                                      max_length=max_length,
                                      padding='max_length',
                                      return_tensors='pt')
            
            label = Q['candidate_answers'].index(Q['answer'][0])
            
            S_entities_mask = torch.ones(encoded_dict['input_ids'].squeeze().shape)*padding_id
            for i,entity in enumerate(entities):
                for NP in entity:
                    if 'bert-base-cased' == model_name:
                        token_ids = tokenizer.encode(NP, add_special_tokens=False, return_tensors='pt')
                        # print(token_ids)
                        S_entities_mask = find_ids(encoded_dict['input_ids'].squeeze(), token_ids.reshape(-1), S_entities_mask, i)
                    elif 'roberta-base' == model_name:
                        token_ids_v1 = tokenizer.encode(NP, add_special_tokens=False, return_tensors='pt')
                        token_ids_v2 = tokenizer.encode(" "+NP, add_special_tokens=False, return_tensors='pt')
                        S_entities_mask = find_ids(encoded_dict['input_ids'].squeeze(), token_ids_v1.reshape(-1), S_entities_mask, i)
                        S_entities_mask = find_ids(encoded_dict['input_ids'].squeeze(), token_ids_v2.reshape(-1), S_entities_mask, i)
                    else:
                        token_ids = tokenizer.encode(NP, add_special_tokens=False, return_tensors='pt')
                        S_entities_mask = find_ids(encoded_dict['input_ids'].squeeze(), token_ids.reshape(-1), S_entities_mask, i)

                
            segment_ids = [padding_id]
            segment_ids.extend([1]*len(tokenizer.encode(S.lower(), add_special_tokens=False)))
            segment_ids.extend([padding_id]*(tokenizer.num_special_tokens_to_add(pair=True)//2))
            segment_ids.extend([2]*len(tokenizer.encode(Q['question'].lower(), add_special_tokens=False)))
            #pad token_type_ids
            if len(segment_ids)<max_length:
                segment_ids.extend([padding_id]*(max_length-len(segment_ids)))
            
            
            edge_index, node1, node2 = [], [], []
            edge_tuples = []
            traversed_entities = []
            for sent in nltk.sent_tokenize(S.lower()):
                sent_entities = []                           
                    
                for i,entity in enumerate(entities):
                    for NP in entity:
                        if NP in sent:
                            sent_entities.append(i)
                            traversed_entities.append(entity)
                            break
                
                sent_entities = list(set(sent_entities))
                if len(sent_entities)>1:
                    # print(sent_entities)
                    for idx in range(len(sent_entities)-1):
                        for idx1 in range(idx+1,len(sent_entities)):
                            if (sent_entities[idx], sent_entities[idx1]) not in edge_tuples:
                                edge_tuples.append((sent_entities[idx], sent_entities[idx1]))
                                edge_tuples.append((sent_entities[idx1], sent_entities[idx]))
            #add self-loops
            for i,entity in enumerate(entities):
                node1.append(i)
                node2.append(i)
            for edge_tuple in edge_tuples:
                node1.append(edge_tuple[0])
                node2.append(edge_tuple[1])
            #pad edge_index
            node1.extend([padding_id]*(300-len(node1)))
            node2.extend([padding_id]*(300-len(node2)))
            edge_index.append(node1)
            edge_index.append(node2)
            
            # idx=0
            # while(1):
            #     boolean_idx = torch.eq(S_entities_mask, idx)
            #     if boolean_idx.sum()==0:
            #         break
            #     idx+=1
            
            # if idx!=len(entities):
            #     print(S)
            #     print(entities)
            #     print(NP_set)
            #     print(S_entities_mask)
            #     print(torch.tensor(edge_index), torch.tensor(edge_index).shape)
            #     cnt+=1
            # print(S)
            # print(entities)
            # print(S_entities_mask)
            # print(torch.tensor(edge_index), torch.tensor(edge_index).shape)
            # print(torch.tensor(segment_ids))
            # print(label)
            tensor_dataset.append((encoded_dict['input_ids'].squeeze(),
                                   encoded_dict['attention_mask'].squeeze(),
                                   encoded_dict['token_type_ids'].squeeze(),
                                   S_entities_mask,
                                   torch.tensor(edge_index),
                                   torch.tensor(segment_ids),
                                   label))
            
    # print(cnt)
    return tensor_dataset
    



def get_tensor_dataset_baseline_stepgame(df,tokenizer,labels_set):
    dataset = []
    max_length=400
    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Tokenizing data'):
        S, Q, y = row['story'], row['question'], row['label']
        label = labels_set.index(y)
        encoded_dict = tokenizer(' '.join(S),
                                 Q,
                                return_token_type_ids=True,
                                max_length=max_length,
                                padding='max_length',
                                return_tensors='pt')
        
        dataset.append((encoded_dict['input_ids'].squeeze(),
                        encoded_dict['attention_mask'].squeeze(),
                        encoded_dict['token_type_ids'].squeeze(),
                        label))
        
    return dataset





def get_tensor_dataset_baseline_spartun_resq(dataset, tokenizer, question_type:str):
    tensor_dataset = []
    max_length = 400
    for i,instances in tqdm(enumerate(dataset['data']), total=len(dataset['data']), desc='Tokenizing data'):
        S = instances['story'][0]
        for Q in instances['questions']:
            encoded_dict = tokenizer(S.lower(),
                                     Q['question'].lower(),
                                     return_token_type_ids=True,
                                     max_length=max_length,
                                     padding='max_length',
                                     return_tensors='pt')
            if Q['q_type']==question_type:
                if question_type=='YN':
                    label = Q['candidate_answers'].index(Q['answer'][0])
                elif question_type=='FR':
                    label = torch.zeros(len(Q['candidate_answers']))
                    for answer in Q['answer']:
                        label[Q['candidate_answers'].index(answer.lower())] = 1
        
                tensor_dataset.append((encoded_dict['input_ids'].squeeze(),
                                       encoded_dict['attention_mask'].squeeze(),
                                       encoded_dict['token_type_ids'].squeeze(),
                                       label))            
            
    return tensor_dataset





def tokenize(sentence):
    """
    Tokenize a string by splitting on non-word characters and stripping whitespace.
    """
    # stemmer = PorterStemmer()
    # return [stemmer.stem(word) for word in tokens]
    return nltk.word_tokenize(sentence)






if __name__=="__main__":
    
    # dataset_used = 'SPARTUN'
    dataset_used = 'ReSQ'
    # dataset_used = 'StepGame-main'
    # base_dir = f'D:/Learning_materials/PhD/Multi-hop_spatial_reasoning_TPR/{dataset_used}'
    base_dir = f'/projdata11/info_fil/sli/Multi-hop_spatial_reasoning_TPR/{dataset_used}'
    model_name = "bert-base-cased"
    # model_name = 'roberta-base'
    # model_name = 'albert-base-v2'
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # dftrain = pd.read_json(f'{base_dir}/Dataset/TrainVersion/train.json')
    # dftrain = dftrain.swapaxes(0,1)
    # dftrain = dftrain.dropna()
    # labels_set = list(set(dftrain['label'].to_list()))
    # print(f'labels_set:{labels_set}')
    
    # print(dftrain.at[49999,'story']) #list
    # print(dftrain.at[49999,'question']) #str
    # print(dftrain.at[49999,'label']) #str
    
    
    if dataset_used == 'ReSQ':
        with open(f'{base_dir}/train_resq.json','r') as f:
            dataset = json.load(f)
    else:
        with open(f'{base_dir}/train.json','r') as f:
            dataset = json.load(f)
    elem=10
    print(list(dataset['data'][elem].keys()))
    
    print(dataset['data'][elem]['story'])
    print(dataset['data'][elem]['questions'])
    # print(dataset['data'][elem]['story_triplets'])
    # print(dataset['data'][elem]['objects_info'])
    
    # for data in dataset['data']:
    #     for questions in data['questions']:
    #         if questions['step_of_reasoning'] == 3:
    #             print(data['story'])
    #             print(questions)

    # story =  dataset['data'][elem]['story'][0]
    # tokens = nltk.word_tokenize(story)
    # tagged = nltk.pos_tag(tokens, tagset='universal')
    # print(tagged)
    # print(nltk.sent_tokenize(story))

        
    
    # grammar = r""""NP:{<ADJ|NOUN>+<VERB>?<NOUN|NUM>}"""   
    # parser = nltk.chunk.regexp.RegexpParser(grammar)
    # tree = parser.parse(tagged)
    # A = []
    # for t in tree.subtrees(lambda t: t.height() < tree.height()):
    #     t_list = list(t)
    #     A.extend(t_list)
    # print(A)    
        
        
    # FR = get_tensor_dataset_baseline_spartun(dataset, tokenizer, question_type='FR')
    # YN = get_tensor_dataset_baseline_spartun(dataset, tokenizer, question_type='YN')
    # print(len(FR), len(YN))
    
    
    # grammar = r""""NP:{<ADJ|NOUN>?<NOUN|NUM>+}"""  
    # parser = nltk.chunk.regexp.RegexpParser(grammar)
    # tree = parser.parse(tagged)
    # for t in tree.subtrees(lambda t: t.height() < tree.height()):
    #     print(list(t))

    # get_tensor_dataset_resq(dataset, tokenizer, model_name=model_name)