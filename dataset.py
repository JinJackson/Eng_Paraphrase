from torch.utils.data import Dataset, DataLoader
import torch
from parser import args
import codecs, json
import numpy as np

def getLabel(num):
  label = [0, 1]
  if num == 0:
      label = [1, 0]
  return label

# 根据bert所需数据格式处理Data
class TrainDataBert(Dataset):
    def __init__(self, train_file, max_length, tokenizer, pair=True):
        self.max_length = max_length
        self.tokenizer = tokenizer

        # 读取train.tsv 和 document_passage.json
        self.pairs = []
        self.deps = []
        train_files = train_file.split(',')

        for train_f in train_files:
          with codecs.open(train_f, 'r', 'utf8') as reader:
              for line in reader:
                  if not line.strip(): continue
                  # info: 1/0 question answer
                  info = line.strip().lower().split('\t')
                  self.pairs.append((info[0], info[1], int(info[2])))

        

        
    def __getitem__(self, index):
        data = self.pairs[index]
        query = data[0]
        answer = data[1]
        max_length = self.max_length

        tokenized_dic = self.tokenizer(text=query, 
                                       text_pair=answer,
                                       max_length=max_length,
                                       padding='max_length',
                                       truncation=True)

        return np.array(tokenized_dic['input_ids']), np.array(tokenized_dic['token_type_ids'] if 'token_type_ids' in tokenized_dic else []), np.array(tokenized_dic['attention_mask']), np.array([data[2]])

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.pairs)

class EvalDataBert(Dataset):
    def __init__(self, eval_file, max_length, tokenizer):
        self.max_length = max_length
        self.tokenizer = tokenizer

        self.deps = []
        self.pairs = []
        self.docs = {}
        self.docs_keys = []
        eval_files = eval_file.split(',')
        for eval_f in eval_files:
          with codecs.open(eval_f, 'r', 'utf8') as reader:
              for line in reader:
                  # info: 1/0 question answer
                  info = line.strip().lower().split('\t')
                  if info[0] not in self.docs:
                      self.docs[info[0]] = []
                      self.docs_keys.append(info[0])
                  self.docs[info[0]].append(info[1]) 
                  if len(info) == 2:
                    self.pairs.append((info[0], info[1], 1))
                  else:
                    self.pairs.append((info[0], info[1], int(info[2])))
                
    def __getitem__(self, index):
        
        data = self.pairs[index]
        query = data[0]
        answer = data[1]
        max_length = self.max_length

        tokenized_dic = self.tokenizer(text=query, 
                                       text_pair=answer, 
                                       max_length=max_length,
                                       padding='max_length',
                                       truncation=True)

        return np.array(tokenized_dic['input_ids']), np.array(tokenized_dic['token_type_ids'] if 'token_type_ids' in tokenized_dic else []), np.array(tokenized_dic['attention_mask']), np.array([data[2]])

    
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.pairs)

class TrainDataRDrop(Dataset):
    def __init__(self, train_file, max_length, tokenizer, pair=True):
        self.max_length = max_length
        self.tokenizer = tokenizer

        # 读取train.tsv 和 document_passage.json
        self.pairs = []
        self.deps = []
        train_files = train_file.split(',')

        for train_f in train_files:
          with codecs.open(train_f, 'r', 'utf8') as reader:
              for line in reader:
                  if not line.strip(): continue
                  # info: 1/0 question answer
                  info = line.strip().lower().split('\t')
                  self.pairs.append((info[0], info[1], int(info[2]), info[3]))
        
        
    def __getitem__(self, index):
        data = self.pairs[index]
        query1 = data[0]
        query2 = data[1]
        aug_query1 = data[3]
        max_length = self.max_length

        tokenized_dic = self.tokenizer(text=query1, 
                                       text_pair=query2,
                                       max_length=max_length,
                                       padding='max_length',
                                       truncation=True)

        tokenized_dic2 = self.tokenizer(text=aug_query1, 
                                       text_pair=query2,
                                       max_length=max_length,
                                       padding='max_length',
                                       truncation=True)

        return np.array(tokenized_dic['input_ids']), np.array(tokenized_dic['token_type_ids'] if 'token_type_ids' in tokenized_dic else []), np.array(tokenized_dic['attention_mask']), np.array([data[2]]), \
               np.array(tokenized_dic2['input_ids']), np.array(tokenized_dic2['token_type_ids'] if 'token_type_ids' in tokenized_dic2 else []), np.array(tokenized_dic2['attention_mask'])

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.pairs)