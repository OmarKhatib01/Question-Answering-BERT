from transformers import AutoTokenizer, BertModel, GPT2LMHeadModel, GPT2Tokenizer
import torch.optim as optim

import torch
import math
import time
import sys
import json
import numpy as np


def main():  
    torch.manual_seed(0)
    answers = ['A','B','C','D']

    train = []
    test = []
    valid = []
    
    file_name = 'train_complete.jsonl'        
    with open(file_name) as json_file:
        json_list = list(json_file)
    for i in range(len(json_list)):
        json_str = json_list[i]
        result = json.loads(json_str)
        
        base = result['fact1'] + ' [SEP] ' + result['question']['stem']
        ans = answers.index(result['answerKey'])
        
        obs = []
        for j in range(4):
            text = base + result['question']['choices'][j]['text'] + ' [SEP]'
            if j == ans:
                label = 1
            else:
                label = 0
            obs.append([text,label])
        train.append(obs)
        
        print(obs)
        print(' ')
        
        # print(result['question']['stem'])
        # print(' ',result['question']['choices'][0]['label'],result['question']['choices'][0]['text'])
        # print(' ',result['question']['choices'][1]['label'],result['question']['choices'][1]['text'])
        # print(' ',result['question']['choices'][2]['label'],result['question']['choices'][2]['text'])
        # print(' ',result['question']['choices'][3]['label'],result['question']['choices'][3]['text'])
        # print('  Fact: ',result['fact1'])
        # print('  Answer: ',result['answerKey'])
        # print('  ')
                
    file_name = 'dev_complete.jsonl'        
    with open(file_name) as json_file:
        json_list = list(json_file)
    for i in range(len(json_list)):
        json_str = json_list[i]
        result = json.loads(json_str)
        
        base = result['fact1'] + ' [SEP] ' + result['question']['stem']
        ans = answers.index(result['answerKey'])
        
        obs = []
        for j in range(4):
            text = base + result['question']['choices'][j]['text'] + ' [SEP]'
            if j == ans:
                label = 1
            else:
                label = 0
            obs.append([text,label])
        valid.append(obs)
        
    file_name = 'test_complete.jsonl'        
    with open(file_name) as json_file:
        json_list = list(json_file)
    for i in range(len(json_list)):
        json_str = json_list[i]
        result = json.loads(json_str)
        
        base = result['fact1'] + ' [SEP] ' + result['question']['stem']
        ans = answers.index(result['answerKey'])
        
        obs = []
        for j in range(4):
            text = base + result['question']['choices'][j]['text'] + ' [SEP]'
            if j == ans:
                label = 1
            else:
                label = 0
            obs.append([text,label])
        test.append(obs)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    optimizer = optim.Adam(model.parameters(), lr=3e-5)
    linear = torch.rand(768,2)

    # model = model.cuda()
    # linear = linear.cuda()

    train_classification(model, linear, train, tokenizer, optimizer, mode='train')
    train_classification(model, linear, valid, tokenizer, optimizer, mode='valid')
    

def train_classification(model, linear, data, tokenizer, optimizer, mode='train'):

    # traning + validation loop
    for epoch in range(10):
        print('Epoch: ',epoch)
        if mode == 'train':
            model.train()
        else:
            model.eval()
            # for validation, mode = 'test'
            correct = 0
            total = 0
        
        for i in range(len(data)):
            obs = data[i]
            text = [x[0] for x in obs]
            labels = torch.tensor([x[1] for x in obs])
            # mask is matrix of (num_questions, classes) where 1 means that the question has that class as label
            mask = torch.zeros(len(obs), 2)
            for j in range(len(obs)):
                mask[j, labels[j]] = 1

            # labels = labels.cuda()
            # mask = mask.cuda()

            inputs = tokenizer(text, padding='max_length', max_length=256, truncation=True, return_tensors="pt")
            # inputs['input_ids'] = inputs['input_ids'].cuda()
            # inputs['attention_mask'] = inputs['attention_mask'].cuda()
            # inputs['token_type_ids'] = inputs['token_type_ids'].cuda()

            optimizer.zero_grad()
            outputs = model(**inputs)

            last_hidden_states = outputs[0]
            last_hidden_states = last_hidden_states[:,0,:]
            logits = torch.matmul(last_hidden_states,linear)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            
            if mode == 'train':
                loss.backward()
                optimizer.step()
                if i % 100 == 0:
                    print('  ',i,loss.item())
            else:
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                print(correct, total)
                print('  Valid: ',correct/total)

# def train_generation(model, linear, data, tokenizer, optimizer, mode='train'):





    
#    Add code to fine-tune and test your MCQA classifier.
           
                 
if __name__ == "__main__":
    main()
