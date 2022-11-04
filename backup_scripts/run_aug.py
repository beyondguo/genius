from transformers import pipeline
from rake_nltk import Rake
from keybert import KeyBERT
import yake
import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stopwords = stopwords.words('english')

def remove_stopwords(text):
    words = word_tokenize(text)
    remain = [w for w in words if w not in stopwords]
    return ' '.join(remain)

import os
DEVICE=int(os.getenv("DEVICE"))
dataset_name = os.getenv("DATASET")
n_aug = int(os.getenv("NAUG"))
max_length = 512
save_name = f'mix{n_aug}_doc_aspect_avg_with_label'

class K2T_Generator:
    def __init__(self, model, device=DEVICE):
        self.generator = pipeline("text2text-generation", model=model, device=device)
    def k2t(self, inputs, beam=1, sample=False, max_length=max_length):
        """
        inputs: text or a list of text
        """
        res = self.generator(inputs, num_beams=beam, do_sample=sample, num_return_sequences=1, max_length=max_length)
        return [item['generated_text'] for item in res]



# keywords extraction:
ke_rake = Rake()  # .extract_keywords_from_text(text).get_ranked_phrases()
ke_yake = yake.KeywordExtractor(n=2,top=10) # n for max_ngram, use .extract_keywords(text)
ke_bert = KeyBERT()  # .extract_keywords(text)

def extract_keywords(text,tool='yake',topk=10):
    assert tool in ['yake','rake','keybert'], f'tool: {tool} not supported'
    if tool == 'rake': # fastest
        ke_rake.extract_keywords_from_text(text)
        kws =  ke_rake.get_ranked_phrases()[:topk]
    if tool == 'yake': # middle
        kws_paris = ke_yake.extract_keywords(text) 
        kws =  [pair[0] for pair in kws_paris]
    if tool == 'keybert': # slowest
        kws_paris = ke_bert.extract_keywords(text,
        keyphrase_ngram_range=(1, 2),top_n=topk)
        kws =  [pair[0] for pair in kws_paris]
    return ' '.join(kws)

def extract_keywords_with_given_words(text,tool='yake',topk=10, given_words=[]):
    assert tool in ['yake','rake','keybert'], f'tool: {tool} not supported'
    if tool == 'rake': # fastest
        ke_rake.extract_keywords_from_text(text)
        kws =  ke_rake.get_ranked_phrases()[:topk]
    if tool == 'yake': # middle
        kws_paris = ke_yake.extract_keywords(text) 
        kws =  [pair[0] for pair in kws_paris]
    if tool == 'keybert': # slowest
        kws_paris = ke_bert.extract_keywords(text,
        keyphrase_ngram_range=(1, 2),top_n=topk)
        kws =  [pair[0] for pair in kws_paris]
    l = len(kws)
    for w in given_words:
        random_idx = random.randrange(l)
        kws.insert(random_idx, w)
    return ' '.join(kws)

from aspect_keybert import AspectKeyBERT
akb = AspectKeyBERT()
def extract_aspect_keywords(text, aspect_keywords, aspect_as_doc=True, topk=10, n_gram=2, add_aspect_keywords=0, shuffle=False, return_str=True):
    # 如果句子太短，抽出来的关键词去生成的质量会很差
    # length = len(text.split(' '))
    # if length < 10:
    #     return remove_stopwords(text).strip() # 居然末尾有空格会严重影响生成效果
        
    kws_paris = akb.extract_aspect_keywords(text, use_aspect_as_doc_embedding=aspect_as_doc, keyphrase_ngram_range=(1,n_gram),\
                                      top_n=topk, aspect_keywords=aspect_keywords)
    kws =  [pair[0] for pair in kws_paris]
    l = len(kws)
    if add_aspect_keywords:
        # add_aspect_keywords可以为1,2,3... 指定要插入几个词，就随机挑选几个
        assert add_aspect_keywords <= len(aspect_keywords)
        candidates = random.sample(aspect_keywords,add_aspect_keywords)
        for w in candidates:
            random_idx = random.randrange(l)
            kws.insert(random_idx, w)
    if shuffle:
        random.shuffle(kws)
    if return_str:
        kws = ' '.join(kws)
    return kws


import pandas as pd
data = pd.read_csv(f'clf_data/{dataset_name}/train.csv')
data = data.dropna()
data = data[data.content != ''] # 处理空值问题
contents = list(data.content)
labels = list(data.label)

######################### label name的处理 ##############################
if 'ng' in dataset_name:
    label2name = {0:"alt atheism",
            1:"computer graphics",
            2:"computer os microsoft windows misc",
            3:"computer system ibm pc hardware",
            4:"computer system mac hardware",
            5:"computer windows x",
            6:"misc for sale",
            7:"rec autos auto",
            8:"rec motorcycles",
            9:"rec sport baseball",
            10:"rec sport hockey",
            11:"sci crypt",
            12:"sci electronics",
            13:"sci medicine med",
            14:"sci space universe",
            15:"soc religion christian",
            16:"talk politics guns gun",
            17:"talk politics mideast",
            18:"talk politics misc",
            19:"talk religion misc"}
elif 'bbc' in dataset_name:
    label2name = {label:label for label in labels}
elif 'imdb' in dataset_name in dataset_name:  # 可以试试都加入一个film，不然可能会生成跟电影完全不搭边的内容。
    label2name = {0: "negative bad", 1: "positive good"}
elif 'yahoo' in dataset_name:
    label2name = {0: "Society Culture",
                  1: "Science Mathematics",
                  2: "Health",
                  3: "Education Reference",
                  4: "Computers Internet",
                  5: "Sports",
                  6: "Business Finance",
                  7: "Entertainment Music",
                  8: "Family Relationships",
                  9: "Politics Government"}
elif 'sst' in dataset_name:
    label2name = {0: "negative bad boring terrible sad", 1: "positive good nice happy great interesting"} #

######################### 不同的关键词抽取策略 ##############################
import time
t1 = time.time()
# ① 经典关键词抽取+label
# keywords_list = [extract_keywords_with_given_words(text, 'keybert', len(text.split(' '))//10, given_words=[label2name[label]]*2) for text,label in zip(contents,labels)]

# ② 抽取跟label最相关的词，不管原始文本讲的啥，Aspect Keywords
# keywords_list = [extract_aspect_keywords(text, aspect_keywords=[label2name[label]], aspect_as_doc=True,topk=min(len(text.split(' '))//10,35), shuffle=False) \
#                         for text,label in zip(contents,labels)]

# ③ aspect跟doc平均representation，然后抽取
# topk = max(min(len(text.split(' '))//10,35),3)  # 最少3个词，最多35个，一般为总词数的1/10
# keywords_list = [extract_aspect_keywords(text, aspect_keywords=[label2name[label]], aspect_as_doc=False,topk=min(len(text.split(' '))//10,35), shuffle=False) \
#                         for text,label in zip(contents,labels)]

# ④ aspect跟doc平均representation，然后抽取，然后+label
def cal_topk(text):
    topk = max(min(len(text.split(' '))//10,35),1)  # 最少3个词，最多35个，一般为总词数的1/10
    return topk
keywords_list = [extract_aspect_keywords(text, aspect_keywords=label2name[label].split(' '), aspect_as_doc=False,add_aspect_keywords=1,topk=cal_topk(text), n_gram=3, shuffle=False) \
                        for text,label in zip(contents,labels)]
t2 = time.time()
print(f'>>>device {DEVICE} extracting keywords cost time: {t2-t1} s')


model = f'saved_models/bart-large-cnn-k2t-wikitext2-keybert-final'
generator = K2T_Generator(model)



# 通过对keywords进行shuffle而产生更加多样化的样本
import random
def shuffle_keywords(keywords_str):
    words = keywords_str.split(' ')
    return ' '.join(random.sample(words,len(words)))

t1 = time.time()
print(f">>>device {DEVICE} Start Augmenting...")
augmented_contents = []
input_keywords_list = []
# 先把原顺序的加进来
augmented_contents += generator.k2t(keywords_list, max_length=max_length)
input_keywords_list += keywords_list
if n_aug > 1:
    for i in range(n_aug-1):
        shuffled_keywords = [shuffle_keywords(k) for k in keywords_list]
        augmented_contents = augmented_contents + generator.k2t(shuffled_keywords,max_length=max_length)
        input_keywords_list += shuffled_keywords
t2 = time.time()
assert len(input_keywords_list) == len(augmented_contents), 'wrong number!'
print(f">>>device {DEVICE} Finished!")
print(f'>>>device {DEVICE} augmentation cost time: {t2-t1} s')
print(f'>>>device {DEVICE} Num of augmented examples: {len(augmented_contents)}')


######################### Cleaning Pipeline ##############################
import re
from nltk.tokenize import sent_tokenize
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
def remove_last_sentence(text):
    # 当超过三个句子，且最后一个句子不以标点结尾，就移除最后一句
    sents = sent_tokenize(text)
    text = ' '.join(sents)
    if len(sents) > 3:
        if sents[-1][-1] not in ".?!。？！\'\"":
            text = ' '.join(sents[:-1])
    return text
        
def clean_pipeline(text):
    return remove_last_sentence(remove_brakets(remove_special_characters(text)))

cleaned_augmented_contents = [clean_pipeline(text) for text in augmented_contents]



mix_df = pd.DataFrame({'content': contents+cleaned_augmented_contents, 'label': labels*(n_aug+1), 'input_keywords': ['NONE']*len(labels)+input_keywords_list})
mix_df.to_csv(f'clf_data/{dataset_name}/{save_name}.csv')
print(f'>>>device {DEVICE} saved to clf_data/{dataset_name}/{save_name}.csv')
