from transformers import AutoTokenizer, BertModel, GPT2LMHeadModel, GPT2Tokenizer
import torch.optim as optim
import torch
import json
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

class Classify():  
    def __init__(self):
        torch.manual_seed(0)

        # classification training  parameters
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")

        for param in self.model.parameters():
            param.requires_grad = False

        self.linear = torch.nn.Linear(768,2)

        for param in self.linear.parameters():
            param.requires_grad = True

        self.optimizer = optim.Adam(self.linear.parameters(), lr=3e-5)
        print("Model Initialized")

        self.train_set = self.preprocess_data('train')
        self.test_set = self.preprocess_data('test')
        self.valid_set = self.preprocess_data('valid')
    
    def preprocess_data(self, data_type):
        answers = ['A','B','C','D']

        file_name = 'data/' + data_type + '_complete.jsonl'
        data = []
    
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
            data.append(obs)
        return data

    def train(self):
        train_loss = []
        train_accuracy = []
        valid_accuracy = []
        for epoch in range(15):
            print(f"Starting training epoch {epoch}")
            epoch_train_loss, epoch_train_accuracy = self.train_model(self.model, self.linear, self.train_set, self.tokenizer, self.optimizer)
            
            with torch.no_grad():
                print(f"Validating epoch {epoch}")
                epoch_valid_accuracy = self.evaluate_model(self.model, self.linear, self.valid_set, self.tokenizer)
                train_loss.append(np.copy(epoch_train_loss))
                train_accuracy.append(np.copy(epoch_train_accuracy))
                valid_accuracy.append(np.copy(epoch_valid_accuracy))

                if epoch == 0 or (epoch > 0 and valid_accuracy[-1] > valid_accuracy[-2]):
                    print("Saving model...")
                    torch.save(self.linear.state_dict(), 'save/linear.pt')
                pd.DataFrame({'train_loss': train_loss, 'train_accuracy': train_accuracy, 'valid_accuracy': valid_accuracy}).to_csv('save/results.csv', index=False)
            
            print(f'Epoch: {epoch} complete | Train Loss: {train_loss[-1]} | Train Accuracy: {train_accuracy[-1]} | Valid Accuracy: {valid_accuracy[-1]}')

    def test(self):
        ## Load weights and test model
        test_model = BertModel.from_pretrained("bert-base-uncased")
        test_linear = torch.nn.Linear(768,2)
        test_linear.load_state_dict(torch.load('save/linear.pt'))
        test_accuracy = self.evaluate_model(test_model, test_linear, self.test_set, self.tokenizer)
        valid_accuracy = self.evaluate_model(test_model, test_linear, self.valid_set, self.tokenizer)
        return valid_accuracy, test_accuracy

    def evaluate_model(self, model, linear, data, tokenizer):
        model.eval()
        total = 0
        correct = 0

        with torch.no_grad():
            for i in range(len(data)):
                obs = data[i]
                text = [x[0] for x in obs]
                labels = torch.tensor([x[1] for x in obs])

                inputs = tokenizer(text, padding='max_length', max_length=256, truncation=True, return_tensors="pt")
                outputs = model(**inputs)

                last_hidden = outputs.last_hidden_state[:,0,:]
                logits = linear(last_hidden)
    
                probs = logits.softmax(dim=1)
                maxind_pred = torch.argmax(probs, dim=0)[1]
                maxind_true = torch.argmax(labels, dim=0)
                
                if maxind_pred == maxind_true:
                    correct += 1
                total += 1

            return correct / total

    def train_model(self, model, linear, data, tokenizer, optimizer):
        model.train()

        # for calculating avg accuracy every N iterations
        total_epoch_iters = 0
        correct_preds = 0
        interval_correct = 0

        # for calculating avg loss every N iterations
        total_epoch_loss = 0
        interval_loss = 0
        interval_iters = 0

        for i in range(len(data)):
            
            obs = data[i]
            text = [x[0] for x in obs]
            labels = torch.tensor([x[1] for x in obs])

            inputs = tokenizer(text, padding='max_length', max_length=256, truncation=True, return_tensors="pt")
            
            optimizer.zero_grad()
            outputs = model(**inputs)

            last_hidden = outputs.last_hidden_state[:,0,:]

            logits = linear(last_hidden)

            loss = torch.nn.functional.cross_entropy(logits, labels)
            total_epoch_loss += loss

            loss.backward()
            optimizer.step()
            
            interval_loss += loss.item()
            interval_iters += 1

            probs = logits.softmax(dim=1)
            maxind_pred = torch.argmax(probs, dim=0)[1]
            maxind_true = torch.argmax(labels, dim=0)

            if maxind_pred == maxind_true:
                interval_correct +=1
                correct_preds += 1

            if i % 100 == 0 or i == len(data)-1:
                print(f"Iter: {i} | Loss: {interval_loss/interval_iters} | Accuracy: {interval_correct/interval_iters}")
                interval_iters = 0
                interval_loss = 0
                interval_correct = 0

            # if i == 10:
            #     print(f"Pred: {maxind_pred}, True: {maxind_true}\n{probs}")

            total_epoch_iters += 1
        
        metrics = total_epoch_loss/total_epoch_iters, correct_preds/total_epoch_iters
        return metrics


class Generate():
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')

        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-5)
        self.loss = torch.nn.CrossEntropyLoss()

        self.label_inds = {'A':32, 'B':33, 'C':34, 'D':35}
        self.labels = ['A', 'B', 'C', 'D']

        self.train_set = self.preprocess_data('train')
        self.valid_set = self.preprocess_data('valid')
        self.test_set = self.preprocess_data('test')

    def preprocess_data(self, data_type):
        file_name = 'data/' + data_type + '_complete.jsonl'
        data = []
    
        with open(file_name) as json_file:
            json_list = list(json_file)
        for i in range(len(json_list)):
            json_str = json_list[i]
            result = json.loads(json_str)
            
            obs = ''
            base = result['fact1'] + ' [SEP] ' + result['question']['stem'] + ' [SEP] '
            ans = result['answerKey']
            choices = ''
            for j in range(len(result['question']['choices'])):
                choices = choices + result['question']['choices'][j]['label'] + ' ' + result['question']['choices'][j]['text'] + ' '

            obs = base + choices + '[ANSWER]' + ans
            data.append(obs)
        return data

    def preprocess_data_text_file(self, data_type):
        file_name = 'data/' + data_type + '_complete.jsonl'
        data = ''
    
        with open(file_name) as json_file:
            json_list = list(json_file)
        for i in range(len(json_list)):
            json_str = json_list[i]
            result = json.loads(json_str)
            
            obs = ''
            base = result['fact1'] + ' [SEP] ' + result['question']['stem'] + ' [SEP] '
            ans = result['answerKey']
            choices = ''
            for j in range(len(result['question']['choices'])):
                choices = choices + result['question']['choices'][j]['label'] + ' ' + result['question']['choices'][j]['text'] + ' '

            obs =  base + choices + '[ANSWER]' + ans + ' <|endoftext|> '
            data = data + obs

        with open(f'{data_type}.txt', 'w') as f:
            f.write(data)

    def train(self):
        # self.preprocess_data_text_file('train')

        command = \
            "python3 -u models/transformers/examples/pytorch/language-modeling/run_clm.py \
            --model_name_or_path gpt2 \
            --train_file data/train.txt \
            --do_train \
            --output_dir models/gpt497 \
            --per_device_train_batch_size 2 \
            --num_train_epochs 5\
            >& save/output.log"
        os.system(command)

        # command = \
        #     "python3 -u models/transformers/examples/pytorch/question-answering/run_qa.py \
        #     --model_name_or_path bert-base-uncased \
        #     --train_file data/train.txt \
        #     --do_train \
        #     --output_dir models/gpt497-gen \
        #     --per_device_train_batch_size 12 \
        #     --learning_rate 3e-5 \
        #     --max_seq_length 384 \
        #     --doc_stride 128 \
        #     --num_train_epochs 2\
        #     >& save/output.log"
        # os.system(command)
  

        # for i in range(1):
        #     print(f"Epoch {i}")
        #     print(f"Validation accuracy: {self.evaluate_model(self.model, self.valid_set, self.tokenizer)}")


        # train_loss = []
        # train_accuracy = []
        # valid_accuracy = []
        # for epoch in range(1):     # 15
        #     print(f"Starting training epoch {epoch}")
        #     # epoch_train_loss, epoch_train_accuracy = self.train_generation(self.model, self.train_set, self.tokenizer, self.optimizer)
        #     epoch_train_loss = self.train_generation(self.model, self.train_set, self.tokenizer, self.optimizer)
            
        #     with torch.no_grad():
        #         print(f"Validating epoch {epoch}")
        #         # epoch_valid_accuracy = self.evaluate_model(self.model, self.linear, self.valid_set, self.tokenizer)
        #         train_loss.append(np.copy(epoch_train_loss))
        #         # train_accuracy.append(np.copy(epoch_train_accuracy))
        #         # valid_accuracy.append(np.copy(epoch_valid_accuracy))

        #         # if epoch == 0 or (epoch > 0 and valid_accuracy[-1] > valid_accuracy[-2]):
        #         #     print("Saving model...")
        #         #     torch.save(self.linear.state_dict(), 'save/gen_linear.pt')
        #         # pd.DataFrame({'train_loss': train_loss, 'train_accuracy': train_accuracy, 'valid_accuracy': valid_accuracy}).to_csv('save/gen_results.csv', index=False)
        #         pd.DataFrame({'train_loss': train_loss, 'train_accuracy': train_accuracy}).to_csv('save/gen_results.csv', index=False)

            
        #     # print(f'Epoch: {epoch} complete | Train Loss: {train_loss[-1]} | Train Accuracy: {train_accuracy[-1]} | Valid Accuracy: {valid_accuracy[-1]}')
        #     print(f'Epoch: {epoch} complete | Train Loss: {train_loss[-1]} | Train Accuracy: {train_accuracy[-1]}')


    def evaluate(self, model, data, tokenizer):
        model.eval()

        with torch.no_grad():

            # for calculating avg accuracy every N iterations
            correct_preds = 0
            for i in range(len(data)):  # len(data)
                obs = data[i]
                inputs = tokenizer(obs[:-1], truncation=True, return_tensors="pt")
                label = tokenizer(obs[-1], truncation=True, return_tensors="pt")['input_ids'][0][0]

                outputs = model(**inputs)
                pred_logits = outputs.logits[0][-1]
                vocab_probs = torch.softmax(pred_logits, dim=0)
                label_probs = [vocab_probs[self.label_inds[key]] for key in self.label_inds.keys()]

                pred_label_ind = torch.argmax(torch.tensor(label_probs), dim=0)
                pred_label = torch.tensor(list(self.label_inds.values()))[pred_label_ind]

                if pred_label == label:
                    correct_preds += 1

            accuracy = correct_preds/len(data)
            return accuracy
        
    def test(self):
        saved_model = GPT2LMHeadModel.from_pretrained('models/gpt497')
        valid_acc = self.evaluate(saved_model, self.valid_set, self.tokenizer)
        test_acc = self.evaluate(saved_model, self.test_set, self.tokenizer)

        return valid_acc, test_acc

    # Attempt 1
    # def train_generation(self, model, data, tokenizer, optimizer, mode='train'):
        # generation training parameters
        # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # model = GPT2LMHeadModel.from_pretrained('gpt2')
        # linear = nn.Linear(768, 4)
        # optimizer = optim.Adam(model.parameters(), lr=3e-5)
        # loss_fn = torch.nn.CrossEntropyLoss()

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = model.to(device)
        # linear = linear.to(device)

        if mode == 'train':
            model.train()
        else:
            model.eval()

        losses = []
        accuracies = []

        loss = torch.zeros(1, requires_grad=True)
        

        for i in range(len(data)):   # len(data)
            obs = data[i]
            inputs = tokenizer(obs[:-1], truncation=True, return_tensors="pt")
            # inputs = tokenizer.encode(''.join(obs[:-1]), truncation=True, return_tensors="pt")
            label = tokenizer(obs[-1], truncation=True, return_tensors="pt")['input_ids'][0][0]
            # label = tokenizer.encode(''.join(obs[-1]), truncation=True, return_tensors="pt")

            inputs = inputs.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            # outputs = model(**inputs)
            outputs = model.generate(**inputs, max_length=len(inputs['input_ids'][0])+1, 
                                     return_dict_in_generate=True, output_scores=True)
            
            # beam_output = model.generate(inputs, max_length=len(inputs[0])+1, 
            #                              num_beams=5, early_stopping=True)

            scores = outputs.scores[0]

            vocab_probs = torch.softmax(scores, dim=1)
            
            # label_probs = [vocab_probs[0][self.label_inds[key]] for key in self.label_inds.keys()]

            # pred_label_ind = torch.argmax(torch.tensor(label_probs), dim=0)
            # pred_label = torch.tensor(list(self.label_inds.values()))[pred_label_ind]
            
            loss = self.loss(vocab_probs, torch.tensor([label], requires_grad = True))
            # loss = Variable(loss, requires_grad = True)
            # loss.requires_grad = True
            
            
            print(loss)
            print('===============')

            loss.backward()
            optimizer.step()

            # linear_input = outputs.last_hidden_state.mean(dim=1) # average across sequence length
            # linear_input = linear_input.to(device)

            # linear_output = linear(linear_input)
            # loss = loss_fn(linear_output.unsqueeze(0), label.unsqueeze(0))

            # if mode == 'train':
            #     optimizer.zero_grad()
            #     loss.backward()
            #     optimizer.step()

            losses.append(loss.item())

        #     if pred_label == label:
        #         accuracies.append(1)
        #     else:
        #         accuracies.append(0)

        # accuracy = sum(accuracies)/len(data)

        # return accuracy, losses
        return losses

    # Attempt 2
    # def train_generation(self, model, data, tokenizer, optimizer, mode='train'):
    #     # generation training parameters
    #     # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    #     # model = GPT2LMHeadModel.from_pretrained('gpt2')
    #     # linear = nn.Linear(768, 4)
    #     # optimizer = optim.Adam(model.parameters(), lr=3e-5)
    #     # loss_fn = torch.nn.CrossEntropyLoss()

    #     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #     model = model.to(device)
    #     # linear = linear.to(device)

    #     if mode == 'train':
    #         model.train()
    #     else:
    #         model.eval()

    #     losses = []
    #     accuracies = []

    #     for i in range(3):   # len(data)
    #         obs = data[i]
    #         inputs = tokenizer(obs[:-1], truncation=True, return_tensors="pt")
    #         label = tokenizer(obs[-1], truncation=True, return_tensors="pt")['input_ids'][0][0]

    #         inputs = inputs.to(device)
    #         label = label.to(device)

    #         outputs = model(**inputs)
    #         # outputs = model(inputs, output_hidden_states=True, return_dict=True)
    #         # outputs = model.generate(**inputs, max_length=len(inputs['input_ids'][0]+1))

    #         print(outputs.last_hidden_state[:,0,:])
    #         print("==============")
    #         pred_logits = outputs.logits[0][-1]
    #         # pred_logits = outputs[0][-1].float().unsqueeze(0)
            
    #         vocab_probs = torch.softmax(pred_logits, dim=0)
    #         label_probs = [vocab_probs[self.label_inds[key]] for key in self.label_inds.keys()]

    #         pred_label_ind = torch.argmax(torch.tensor(label_probs), dim=0)
    #         pred_label = torch.tensor(list(self.label_inds.values()))[pred_label_ind]

    #         # linear_input = outputs.last_hidden_state.mean(dim=1) # average across sequence length
    #         # linear_input = linear_input.to(device)

    #         # linear_output = linear(linear_input)
    #         # loss = loss_fn(linear_output.unsqueeze(0), label.unsqueeze(0))
    #         loss = self.loss(pred_label.float().unsqueeze(0), label.float().unsqueeze(0))

    #         # print(obs)
    #         # print('===============')
    #         # print(obs[-1])
    #         # print('===============')
    #         # print(label)
    #         # print('===============')
    #         # print(pred_label)
    #         # print('===============')

    #         if mode == 'train':
    #             optimizer.zero_grad()
    #             # loss.backward()
    #             optimizer.step()

    #         losses.append(loss.item())

    #         if pred_label == label:
    #             accuracies.append(1)
    #         else:
    #             accuracies.append(0)

    #     accuracy = sum(accuracies)/len(data)

    #     return accuracy, losses


    # def zero_shot(self, data):
    #     tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    #     model = GPT2LMHeadModel.from_pretrained('gpt2')
    #     tokenizer.pad_token_id = tokenizer.eos_token_id

    #     for i in range(5): # len(data)
    #         obs = data[i]
    #         inputs = tokenizer(obs[:-1], return_tensors="pt")
    #         # Print the scores for each token generated with Greedy Search
    #         outputs = model.generate(**inputs, max_new_tokens=10, return_dict_in_generate=True, output_scores=True)
    #         # max_length=len(inputs[0])+1

    #         transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
    #         input_length = inputs.input_ids.shape[1]
    #         generated_tokens = outputs.sequences[:, input_length:]

    #         result = {}

    #         for tok, score in zip(generated_tokens[0], transition_scores[0]):
    #             if tokenizer.decode(tok).strip() == 'A':
    #                 # | token | token string | logits | probability
    #                 print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.3f} | {np.exp(score.numpy()):.2%}")


def create_plots():
    # Create plots from results.csv
    df = pd.read_csv('save/results.csv')
    plt.plot(df['train_accuracy'], label='train_acuracy')
    plt.plot(df['valid_accuracy'], label='valid_acuracy')
    plt.legend()
    plt.savefig('save/classifier_train_test_acc.png')
                 
if __name__ == "__main__":
    classifier = Classify()
    # classifier.train()
    valid_acc, test_acc = classifier.test()
    print(f"CLASSIFIER Validation accuracy: {valid_acc} | Test accuracy: {test_acc}")
    create_plots()


    generator = Generate()
    # for training
    # generator.train()

    # for fine-tune validating and testing
    # valid_acc, test_acc = generator.test()
    # print(f"GENERATOR Validation accuracy: {valid_acc} | Test accuracy: {test_acc}")

    # for zero-shot accuracies on the validation and test sets
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer.pad_token_id = tokenizer.eos_token_id

    valid_acc = generator.evaluate(model, generator.valid_set, tokenizer)
    test_acc = generator.evaluate(model, generator.test_set, tokenizer)
    print(f"zero-shot GENERATOR Validation accuracy: {valid_acc} | Test accuracy: {test_acc}")




# def generate(self):
    #     correct = 0
    #     total = 0
    #     for i in range(len()):
    #         obs = self.train_set[i][:-1]
    #         inputs = self.tokenizer(obs, return_tensors="pt")
    #         # outputs = self.model(**inputs, max_length=len(inputs['input_ids'][0]+1))
    #         outputs = self.model(**inputs)
    #         N_token = len(self.mode)
    #         tokens_decoded = [enc.decode([token]) for token in range(N_token)]
    #         print(outputs.logits)
    #         exit()
    #         # N_token = len(enc.encoder)
    #         # tokens_decoded = [enc.decode([token]) for token in range(N_token)]

    #         temp = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    #         tokens = temp.split(' ')
    #         pred_answer = tokens[-1]
    #         print(tokens)
    #         print(pred_answer)
    #         exit()

            # if answer == 
            #     correct += 1
            # total += 1
        #     obs = data[i]
        #     text = [x[0] for x in obs]
        #     labels = torch.tensor([x[1] for x in obs])

        #     inputs = tokenizer(text, padding='max_length', max_length=256, truncation=True, return_tensors="pt")
        #     outputs = model(**inputs)

        #     last_hidden = outputs.last_hidden_state[:,0,:]
        #     logits = linear(last_hidden)

        #     probs = logits.softmax(dim=1)
        #     maxind_pred = torch.argmax(probs, dim=0)[1]
        #     maxind_true = torch.argmax(labels, dim=0)
            
        #     if maxind_pred == maxind_true:
        #         correct += 1
        #     total += 1
        # outputs = model.generate(**inputs, max_length=len(inputs['input_ids'][0]+1))
        # temp = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # tokens = temp.split(' ')
        # answer = tokens[-1]

# def train_generation(model, linear, data, tokenizer, optimizer, mode='train'):
#         generation training parameters
#         tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#         model = GPT2LMHeadModel.from_pretrained('gpt2')
#         optimizer = optim.Adam(model.parameters(), lr=3e-5)
#    Add code to fine-tune and test your MCQA classifier.



