import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import os
import datasets

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

class MyDataloaders:
    """
    读取标准化csv数据集，自动划分验证集。
    初始化参数：train_path, test_path, tokenizer, maxlen, bsz, collate_fn
    对象的重要属性：
    unique_labels
    label2idx
    idx2label
    train_dataloader
    val_dataloader
    test_dataloader
    """
    def __init__(self,train_path, test_path, tokenizer, maxlen, train_bz, eval_bz, collate_fn, split_valid_from=None):
        raw_train_df = pd.read_csv(train_path)  # validation set会从中划分
        train_texts, train_labels = list(raw_train_df['content']), list(raw_train_df['label'])
        self.unique_labels = sorted(list(set(train_labels)))
        if split_valid_from is not None:
            # 指定了验证集应该从训练集的前多少个里面筛选，应用于增强样本被添加在原始样本后面
            orig_train_texts, val_texts, orig_train_labels, val_labels = \
                train_test_split(train_texts[:split_valid_from], train_labels[:split_valid_from], test_size=.2,random_state=123)
            train_texts = orig_train_texts + train_texts[split_valid_from:]
            train_labels = orig_train_labels + train_labels[split_valid_from:]
        else:
            train_texts, val_texts, train_labels, val_labels = \
                train_test_split(train_texts, train_labels, test_size=.2,random_state=123)
                # 保持这里的random_state不变，可使每次加载数据的划分都相同
                # 比方在使用R-Aug时，只要原数据集的顺序保持相同，那么划分训练测试集之后的也相同
        raw_test_df = pd.read_csv(test_path)
        test_texts, test_labels = list(raw_test_df['content']), list(raw_test_df['label'])

        self.label2idx = {self.unique_labels[i]: i for i in range(len(self.unique_labels))}
        self.idx2label = {self.label2idx[label]: label for label in self.label2idx}

        train_dataset = MyDataset(tokenizer, train_texts, train_labels, self.label2idx, maxlen)
        val_dataset = MyDataset(tokenizer, val_texts, val_labels, self.label2idx, maxlen)
        test_dataset = MyDataset(tokenizer, test_texts, test_labels, self.label2idx, maxlen)
        # 前面的train_test_split默认就有shuffle了，所以dataloader这里可以放心地关闭
        self.train_dataloader = DataLoader(train_dataset, batch_size=train_bz, shuffle=False, collate_fn=collate_fn)
        self.val_dataloader = DataLoader(val_dataset, batch_size=eval_bz, collate_fn=collate_fn)
        self.test_dataloader = DataLoader(test_dataset, batch_size=eval_bz, collate_fn=collate_fn)