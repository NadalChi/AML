import torch
from transformers import BertTokenizer, BertModel
import glob
import chinese_converter
from string import punctuation
import ast
import numpy as np
import re

class AMLDataset(torch.utils.data.Dataset):
    def __init__(self, data_path='./data/training_set/', model_path='./pytorch-ernie'):
#         self.paths = glob.glob(data_path+"*.txt")
        self.paths = glob.glob(data_path+"108.txt")+glob.glob(data_path+"209.txt")+glob.glob(data_path+"1.txt")
        self.tokenizer = BertTokenizer.from_pretrained('./pytorch-ernie', do_lower_case = True)
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with open(path, encoding = 'utf-8') as f:
            text = f.read().split('\n')
            target = " ".join(ast.literal_eval(text[0]))
            print(target)
            input_data = "".join(text[1:])
            input_data = re.sub(r'[A-Za-z]','', input_data)
            # remove punctuation
            punctuation_characters = punctuation+'。、·「」！，）：（【】'
            input_data = input_data.translate(str.maketrans('', '',punctuation_characters ))
            # remove space
            input_data = input_data.replace(' ','')
        return input_data, target
        

def convert_batch_token_target(batch_input_data, batch_target):
    batch_input_data = [chinese_converter.to_simplified(d) for d in batch_input_data]
    input_tensor = tokenizer(batch_input_data, padding=True, return_tensors="pt", add_special_tokens=False)
    input_token, padding_mask = input_tensor["input_ids"], input_tensor['attention_mask']
    batch_target = process_target(input_token, batch_target)
    return  input_token, padding_mask, torch.tensor(batch_target)

def process_target(input_token, batch_target):
    targets = []
    for slice_token, names in zip(input_token, batch_target):
        text = tokenizer.decode(slice_token, clean_up_tokenization_spaces=False).split(' ')
        target = [0]*len(text)
        for name in names.split(" "):
            name = chinese_converter.to_simplified(name)
            for i in range(len(text)-len(name)):
                if name == "".join(text[i:i+len(name)]):
                    target[i:i+len(name)] = [1]*len(name)
        targets.append(target)
    return targets
                    

