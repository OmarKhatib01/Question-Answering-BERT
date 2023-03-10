from transformers import AutoTokenizer, BertModel, GPT2LMHeadModel, GPT2Tokenizer
import torch.optim as optim

import torch
import math
import time
import sys
import json
import numpy as np
import pandas as pd


def main():  
    torch.manual_seed(0)
    answers = ['A','B','C','D']

    train = []
    test = []
    valid = []
    
    # making train data 
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
            text = base + ' ' + result['question']['choices'][j]['text'] + ' [SEP]'
            if j == ans:
                label = 1
            else:
                label = 0
            obs.append([text,label])
        train.append(obs)
        
        # print(obs)
        print(' ')

        # print(' ',result['question']['choices'][0]['label'],result['question']['choices'][0]['text'])
        # print(' ',result['question']['choices'][1]['label'],result['question']['choices'][1]['text'])
        # print(' ',result['question']['choices'][2]['label'],result['question']['choices'][2]['text'])
        # print(' ',result['question']['choices'][3]['label'],result['question']['choices'][3]['text'])
        # print('  Fact: ',result['fact1'])
        # print('  Answer: ',result['answerKey'])
        # print('  ')

    # making valid data   
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
        
    # making test data
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
            text = base + ' ' + result['question']['choices'][j]['text'] + ' [SEP]'
            if j == ans:
                label = 1
            else:
                label = 0
            obs.append([text,label])
        test.append(obs)


    # classification training  parameters
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    optimizer = optim.Adam(model.parameters(), lr=3e-5)

    for param in model.parameters():
        param.requires_grad = False

    linear = torch.rand(768,2, requires_grad=True)

    losses = []
    accuracies = []
    for epoch in range(10):
        epoch_loss = train_classification(model, linear, train, tokenizer, optimizer, mode='train')
        
        with torch.no_grad():
            epoch_acc = train_classification(model, linear, valid, tokenizer, optimizer, mode='validate')
            losses.append(np.copy(epoch_loss))
            accuracies.append(np.copy(epoch_acc))

            if epoch == 0 or (epoch > 0 and accuracies[-1] > accuracies[-2]):
                print("Saving model...")
                torch.save(model.state_dict(), 'save/model.pt')
            pd.DataFrame({'loss': losses, 'accuracy': accuracies}).to_csv('save/results.csv', index=False)
        
        print('Epoch: ', epoch, ' complete | Loss: ', losses[-1], ' | Accuracy: ', accuracies[-1])

    saved_model = torch.load('save/model.pt')
    
    # generation training parameters
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # model = GPT2LMHeadModel.from_pretrained('gpt2')
    # optimizer = optim.Adam(model.parameters(), lr=3e-5)
    

def train_classification(model, linear, data, tokenizer, optimizer, mode='train'):
        
    if mode == 'train':
        model.train()
    else: #mode == 'validate'
        model.eval()

    total = 0
    correct = 0
    total_epoch_loss = 0

    for i in range(len(data)):
        
        obs = data[i]
        text = [x[0] for x in obs]
        labels = torch.tensor([x[1] for x in obs])

        inputs = tokenizer(text, padding='max_length', max_length=256, truncation=True, return_tensors="pt")
        
        optimizer.zero_grad()
        outputs = model(**inputs)

        last_hidden = outputs.last_hidden_state[:,0,:]
        logits = torch.matmul(last_hidden,linear)

        loss = torch.nn.functional.cross_entropy(logits, labels)
        total_epoch_loss += loss

        if mode == 'train':
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print("Iter ", i, " | Loss ", loss.item())
        else: #mode == 'validate'
            # get prediction
            probs = logits.softmax(dim=1)
            maxind_pred = torch.argmax(probs, dim=0)[1]
            maxind_true = torch.argmax(labels, dim=0)
            print(f"Pred: {maxind_pred}, True: {maxind_true}")

            if maxind_pred == maxind_true:
                correct += 1
        total += 1
    
    if mode == 'train':
        return total_epoch_loss/total
    else: # mode == 'validate'
        return correct/total

# def train_generation(model, linear, data, tokenizer, optimizer, mode='train'):


    
#    Add code to fine-tune and test your MCQA classifier.
           
                 
if __name__ == "__main__":
    main()



            # mask is matrix of (num_questions, classes) where 1 means that the question has that class as label
            # mask = torch.zeros(len(obs), 2)
            # for j in range(len(obs)):
            #     mask[j, labels[j]] = 1

            # labels = labels.cuda()
            # mask = mask.cuda()

            # inputs['input_ids'] = inputs['input_ids'].cuda()
            # inputs['attention_mask'] = inputs['attention_mask'].cuda()
            # inputs['token_type_ids'] = inputs['token_type_ids'].cuda()

            # logits = torch.matmul(last_hidden, linear)
            # logits = torch.exp(logits)
            # denom = torch.sum(logits, 1) #denom; sum logits over dim 1
            # denom = denom.unsqueeze(1) #unsqueeze - reinflate dim 1
            # numer = logits
            # probs = numer / denom
            # probs = probs * mask
            # probs = torch.sum(probs, 1)
            # log_probs = -1*torch.log(probs)
            # loss2 = torch.sum(log_probs, 0)/4