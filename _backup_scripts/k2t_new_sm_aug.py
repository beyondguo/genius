from transformers import pipeline
from rake_nltk import Rake
from keybert import KeyBERT
import random
random.seed(1)
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import types
from tqdm import tqdm
import yake
import re
from nltk.tokenize import word_tokenize
from aspect_keybert import AspectKeyBERT
akb = AspectKeyBERT()

def keynotes_yake(text, max_ngram=3, topk=20):
    if type(topk) == types.FunctionType: # 使用一个函数来确定topk
        topk_num = topk(text)
    else:
        topk_num = topk
    ke_yake = yake.KeywordExtractor(n=max_ngram,top=topk_num)
    kws_paris = ke_yake.extract_keywords(text) 
    kws =  [pair[0] for pair in kws_paris]
    return kws

def keysents_semantic(text, candidates=None, aspect_keywords=None, max_ngram=3, topk=20, aspect_only=False):
    if type(topk) == types.FunctionType: # 使用一个函数来确定topk
        topk_num = topk(text)
    else:
        topk_num = topk
    kws_paris = akb.extract_aspect_keywords(text,candidates=candidates, top_n=topk_num, keyphrase_ngram_range=(1,max_ngram),
                                      aspect_keywords=aspect_keywords,
                                      use_aspect_as_doc_embedding=aspect_only,)
    kws =  [pair[0] for pair in kws_paris]
    return kws

table = str.maketrans({"-":  r"\-", "]":  r"\]", "[":  r"\[", "\\": r"\\", \
                       "^":  r"\^", "$":  r"\$", "*":  r"\*", ".":  r"\.", \
                        "(":  r"\(", ")":  r"\)", \
                       })

def mask_unimportant_parts(text, kws):
    """
    kws: List

    输出例子：
    '<mask> Google is acquiring Kaggle <mask> hosts data science and machine learning competitions 
    <mask> Google <mask> hosting its Cloud Next conference <mask> San Francisco this week <mask> 
    Kaggle co-founder CEO Anthony Goldbloom declined <mask> Google <mask> Kaggle <mask>'
    """    

    all_ids = []
    for w in kws: # 找出每个词的位置
        try:
            for m in list(re.finditer(w.translate(table),text)): 
                all_ids += list(range(m.start(),m.end()))
        except Exception as e:
            print(e)
            print(w, '||', text)
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


from torch.utils.data import Dataset
class MDataset(Dataset):
    def __init__(self, m_list):
        self.masked_contents = m_list
    def __len__(self):
        return len(self.masked_contents)
    def __getitem__(self, i):
        return self.masked_contents[i]


# 清洁工：
# function to remove special characters
def remove_special_characters(text):
    # 移除非字母、非数字、非主要标点
    pat = r'[^a-zA-z0-9.,!?/:;\"\'\s]' 
    text =  re.sub(pat, '', text)
    text =  re.sub('``', '', text)
    text =  re.sub(r'\s{2,}', ' ', text) # 匹配两个或更多的空白字符
    return text.strip()
def remove_brakets(text):
    text =  re.sub(r'\[(.*)\]', '', text)
    text =  re.sub(r'\((.*)\)', '', text)
    return text
def remove_last_sentence(text):
    # 当超过1个句子，且最后一个句子不以标点结尾，就移除最后一句
    sents = sent_tokenize(text)
    text = ' '.join(sents)
    if len(sents) > 1:
        if sents[-1][-1] not in ".?!。？！\'\"":
            text = ' '.join(sents[:-1])
    return text
        
def clean_pipeline(text):
    return remove_last_sentence((remove_special_characters(remove_brakets(text))))



# K2T生成器
device = 7
model = 'saved_models/bart-large-c4-realnewslike-yake_mask-t_2/checkpoint-689992'  # 13 million
k2t = pipeline("text2text-generation", model=model,device=device)
tokenizer = k2t.tokenizer

from datasets import load_dataset
orig_dataset = load_dataset("cnn_dailymail",'3.0.0')

N_TRAIN = 100
N_AUG = 1
dataset = orig_dataset['train'].select(range(N_TRAIN))

contents = dataset['article']
summaries = dataset['highlights']
avg_len = sum([len(c.split(' ')) for c in contents])/len(contents)
print('avg_content_len: ',avg_len)
print('avg_summary_len: ',sum([len(c.split(' ')) for c in summaries])/len(summaries))


# 由于现在k2t生成的长度限制在了200，对于summarization的原文来说太短了
# 所以这里考虑将原文分成上下两半，分别生成，再拼接起来
m_list = []
for c, s in zip(tqdm(contents), summaries):
    sents = sent_tokenize(c)
    keysents = keysents_semantic(c,candidates=sents, aspect_keywords=[s], topk=3)
    addtional_kws = keysents_semantic(c,aspect_keywords=[s],max_ngram=3, topk=50)
    # for sent in keysents:
    #     print(sents.index(sent))
    m = mask_unimportant_parts(' '.join(sents), keysents+addtional_kws)
    m_list.append(m)

m_dataset = MDataset(m_list)

generated_contents = []
for _ in range(N_AUG):
    for out in tqdm(k2t(m_dataset, num_beams=3, do_sample=True, max_length=500, batch_size=8)):
        gt = out[0]['generated_text']
        gt = clean_pipeline(gt)
        generated_contents.append(gt)

# for i in range(10):
#     print(len(tokenizer(generated_contents[i])['input_ids']))


augmented_dataset = {'article':generated_contents, 'highlights':summaries*(N_AUG)}
import pandas as pd
df = pd.DataFrame(augmented_dataset)
f_name = f'sm_data/cnn_first{N_TRAIN}_aug{N_AUG}.pkl'
df.to_pickle(f_name)
f_name

