import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from collections import defaultdict
from transformers import AutoTokenizer
from datasets import load_dataset
nltk.download('stopwords')
nltk.download('punkt')
import random

# pretrained checkpoint:
# model_checkpoint = 'facebook/bart-large'
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


dataset4 = load_dataset('c4','realnewslike', cache_dir='../saved_datasets/hf_cache')
print(dataset4)
import sys
sys.path.append('..')
import yake
import re

def mask_unimportant_parts(text, max_ngram=3, topk=20):
    """
    输出例子：
    '<mask> Google is acquiring Kaggle <mask> hosts data science and machine learning competitions 
    <mask> Google <mask> hosting its Cloud Next conference <mask> San Francisco this week <mask> 
    Kaggle co-founder CEO Anthony Goldbloom declined <mask> Google <mask> Kaggle <mask>'
    """
    ke_yake = yake.KeywordExtractor(n=max_ngram,top=topk)
    kws_paris = ke_yake.extract_keywords(text) 
    kws =  [pair[0] for pair in kws_paris]

    words_idxs = []
    all_ids = []
    for w in kws: # 找出每个词的位置
        for m in list(re.finditer(w,text)): 
            all_ids += list(range(m.start(),m.end()))
    all_ids = sorted(list(set(all_ids)))
    # 给不连续的部分中间补上mask token
    masked_text = []
    for i,id in enumerate(all_ids):
        if i == 0 and id != 0: # 开头补mask
            masked_text.append('<mask> ')
        if id - all_ids[i-1] > 1: # 说明中间有东西
            masked_text.append(' <mask> ')
        masked_text.append(text[id])
        if i == len(all_ids)-1 and id != len(text)-1: # 最后补mask
            masked_text.append(' <mask>')
    masked_text = ''.join(masked_text)
    return masked_text


#Cleaning Pipeline 
import re
def remove_special_characters(text):
    # 移除非字母、非数字、非主要标点
    pat = r'[^a-zA-z0-9.,!?/:;\"\'\s]' 
    new_text =  re.sub(pat, '', text)
    return new_text
def remove_brakets(text):
    # 移除小括号中括号
    text =  re.sub(r'\[(.*)\]', '', text)
    text =  re.sub(r'\((.*)\)', '', text)
    return text
def clean_pipeline(text):
    return remove_brakets(remove_special_characters(text))


def add_kws_to_dataset_longtext(examples):
    """
    先直接用\n\n给分成很多段落，然后把太短的过滤掉
    接着抽取关键词(总词数的的1/10)
    由于样本增多了，记得在map的时候设置remove_columns=your_dataset['train'].column_names
    """
    res = defaultdict(list)
    documents = examples['text']
    # 针对C4数据集的处理：
    for document in documents:
        document = clean_pipeline(document)
        document = document.replace('\n',' ')
        l = len(document.split(' '))
        # if l > 50 and l < 200:
            # res['text'].append(document)
            # res['keynote'].append(mask_unimportant_parts(document,max_ngram=3, topk=max(l//5,10)))
        if l > 0 and l < 500:
            res['text'].append(document)
            res['keynote'].append(mask_unimportant_parts(document,max_ngram=3, topk=max(l//5,5)))
        elif l >= 500:  # 500以上就先分句，然后取前50个句子
            sents = sent_tokenize(document)
            document = ' '.join(sents[:25])
            res['text'].append(document)
            res['keynote'].append(mask_unimportant_parts(document,max_ngram=3, topk=max(l//5,5)))
        else:
            pass
    return res


random.seed(1)

# 如果只训练一个子集：
# num_doc = 13799838
# # C4 train中总共文档数 13799838
# dataset_with_kws = dataset4['train'].select(random.sample(range(13799838),num_doc))\
#                                 .map(add_kws_to_dataset_longtext,batched=True,\
#                                  remove_columns=dataset4['train'].column_names,\
#                                  batch_size=100,num_proc=500,)nvidia-smi


# 全量数据集：
dataset_with_kws = dataset4.map(add_kws_to_dataset_longtext,batched=True,\
                            remove_columns=dataset4['train'].column_names,\
                            batch_size=100,num_proc=400,)

print(dataset_with_kws)

def show(i):
    print(f">>>text:\n{dataset_with_kws['text'][i]}")
    print(f">>>kws:\n{dataset_with_kws['keynote'][i]}")


# dataset_with_kws.save_to_disk(f'saved_datasets/c4-l_50_200-d_{num_doc}-yake_mask-t_{len(dataset_with_kws)}/')  # 当前实验的版本，390万的长度50~200的文档
# dataset_with_kws.save_to_disk(f'saved_datasets/c4-realnewslike-yake_mask-t_{len(dataset_with_kws)}/')


#-------------------
# import os
# os.system("sh oc.sh")
