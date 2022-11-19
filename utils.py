import os
import torch
import random
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset


def makedir(path):
    is_exist = os.path.exists(path)
    if is_exist:
        return '%s already exists!'%path
    else:
        os.makedirs(path)

def setup_seed(seed):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


class OrderNamespace(argparse.Namespace):
    def __init__(self, **kwargs):
        self.__dict__['order'] = []
        super(OrderNamespace, self).__init__(**kwargs)
    def __setattr__(self,attr,value):
        #  如果没有这个if，args中的str类型会在log中打印两次。
        #  猜测可能是由于父类会对str重复调用__setattr__方法的缘故
        if attr not in self.__dict__['order']:
            self.__dict__['order'].append(attr)
        super(OrderNamespace, self).__setattr__(attr, value)


def fix_seed(i):
    random.seed(i)
    np.random.seed(i)
    torch.manual_seed(i)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(i)



class MyDataset(Dataset):
    def __init__(self, tokenizer, texts, labels, label2idx, maxlen):
        self.tokenizer = tokenizer
        # no padding now，use data_collator for dynamic padding later
        texts = [t if (t != None and str(t) != 'nan') else '' for t in texts]
        self.encodings = tokenizer(texts, truncation=True, max_length=maxlen)
        self.labels = labels
        self.label2idx = label2idx
    def __getitem__(self, idx):
        item = {k:torch.tensor(v[idx]) for k,v in self.encodings.items()}
        item['labels'] = torch.tensor(self.label2idx[self.labels[idx]])  # 'labels' column should contain the idx of label, instead of the label string
        return item
    def __len__(self):
        return len(self.labels)


def get_dataloader(file_path, tokenizer, label2idx, maxlen, bsz, collate_fn, shuffle=True):
    # input a csv file, return a dataloader
    df = pd.read_csv(file_path)
    texts, labels = list(df['content']), list(df['label'])
    dataset = MyDataset(tokenizer, texts, labels, label2idx, maxlen)
    dataloader = DataLoader(dataset, batch_size=bsz, collate_fn=collate_fn, shuffle=shuffle)
    return dataloader

def get_dataloader_from_list(texts, labels, tokenizer, label2idx, maxlen, bsz, collate_fn, shuffle=True):
    dataset = MyDataset(tokenizer, texts, labels, label2idx, maxlen)
    dataloader = DataLoader(dataset, batch_size=bsz, collate_fn=collate_fn, shuffle=shuffle)
    return dataloader
