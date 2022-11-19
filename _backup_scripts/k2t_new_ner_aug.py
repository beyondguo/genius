
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import pandas as pd
import numpy as np
import nltk
# nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize
from more_itertools import locate
from transformers import pipeline
import re
import yake
import types
from collections import defaultdict
from datasets import load_dataset, Dataset
from ner_and_qa.s2t_utils import clean_pipeline, remove_special_characters, get_stopwords, S2T_Dataset
import random
random.seed(5)


dataset_name = 'conll2003'
# dataset_name = 'wikiann'
raw_datasets = load_dataset(dataset_name,'en')


def keynotes_yake(text, max_ngram=3, topk=20):
    if type(topk) == types.FunctionType: # 使用一个函数来确定topk
        topk_num = topk(text)
    else:
        topk_num = topk
    ke_yake = yake.KeywordExtractor(n=max_ngram,top=topk_num)
    kws_paris = ke_yake.extract_keywords(text) 
    kws =  [pair[0] for pair in kws_paris]
    return kws

from aspect_keybert import AspectKeyBERT
akb = AspectKeyBERT()
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

def show_dataset(dataset, i, print_to_file=None):
    words = dataset['tokens'][i]
    labels = dataset['ner_tags'][i]
    line1 = ""
    line2 = ""
    for word, label in zip(words, labels):
        full_label = tag_names[label]
        max_length = max(len(word), len(full_label))
        line1 += word + " " * (max_length - len(word) + 1)
        line2 += full_label + " " * (max_length - len(full_label) + 1)
    if print_to_file is not None:
        print('[%s]--------------------------------------'%i,file=print_to_file)
        print(line1,file=print_to_file)
        print(line2,file=print_to_file)
    else:
        print(line1)
        print(line2)

def show(tokens, tags):
    words = tokens
    labels = tags
    line1 = ""
    line2 = ""
    for word, label in zip(words, labels):
        full_label = tag_names[label]
        max_length = max(len(word), len(full_label))
        line1 += word + " " * (max_length - len(word) + 1)
        line2 += full_label + " " * (max_length - len(full_label) + 1)
    print(line1)
    print(line2)



# 合并多条样本
def concat_multiple_sequences(dataset, size=3, overlap=True):
    # 传入正经的huggingface dataset格式
    # 如果是子集的话，建议使用select方法来筛选
    new_dataset = defaultdict(list)
    l = len(dataset)
    if overlap: # 连续窗口滑动
        for i in range(l-size):
            concat_tokens = np.concatenate(dataset[i:i+size]['tokens'])
            concat_tags = np.concatenate(dataset[i:i+size]['ner_tags'])
            new_dataset['tokens'].append(concat_tokens)
            new_dataset['ner_tags'].append(concat_tags)
    else:  # 互相不重叠
        for i in range(l//size):
            concat_tokens = np.concatenate(dataset[i*size:(i+1)*size]['tokens'])
            concat_tags = np.concatenate(dataset[i*size:(i+1)*size]['ner_tags'])
            new_dataset['tokens'].append(concat_tokens)
            new_dataset['ner_tags'].append(concat_tags)
    return new_dataset


tag_names = raw_datasets['train'].features['ner_tags'].feature.names

def get_mention_name(tag):
    # tag: the number/index of the tag name
    # tag_names: the list of tag names
    # mention: ORG, LOC, etc.
    return tag_names[tag].split('-')[-1]

# 单独把实体抽出来
def extract_mentions(tokens, tags):
    """
    return: 
    mentions: []
    mention_dict: {'MISC': [], 'PER': [], 'LOC': [], 'ORG': []}
    """
    mentions = []
    mention_dict = {t:[] for t in list(set([t.split('-')[-1] for t in tag_names])) if t != 'O'}
    for i in range(len(tokens)):
        mention = get_mention_name(tags[i])
        if mention == 'O':
            continue
        if tags[i] % 2 == 1:
            # the start
            mention_dict[mention].append([tokens[i]])
            mentions.append([tokens[i]])
        else:
            # the remaining part
            mention_dict[mention][-1].append(tokens[i])
            mentions[-1].append(tokens[i])
    for k in mention_dict:
        if mention_dict[k]: # not empty
            mention_dict[k] = [' '.join(items) for items in mention_dict[k]]
    mentions = [' '.join(items) for items in mentions]
    return mentions,mention_dict
    

def get_spans(tokens,window=3):
    spans = []
    for i in range(len(tokens) // window):
        spans.append(' '.join(tokens[i*window:(i+1)*window]))
    return spans

def extract_mention_spans(tokens, tags):
    """
    把一个句子中"""
    text = ' '.join(tokens)
    kws = keynotes_yake(text, 2, 5)
    spans = get_spans(tokens, window=3)
    mentions, _ = extract_mentions(tokens, tags)
    wanted_spans = []
    for span in spans[1:-1]:
        for w in mentions+[]:
            if w in span:
                wanted_spans.append(span)
                break
    m = '<mask> ' + ' <mask> '.join(wanted_spans) + ' <mask>'
    m = f"{spans[0]} {m} {spans[-1]}" # 相当于限制了边界
    return wanted_spans, m

from torch.utils.data import Dataset
class MDataset(Dataset):
    def __init__(self, m_list):
        self.masked_contents = m_list
    def __len__(self):
        return len(self.masked_contents)
    def __getitem__(self, i):
        return self.masked_contents[i]


device = 1
# model = f'saved_models/bart-large-c4-l_50_200-d_13799838-yake_mask-t_3900800/checkpoint-152375' # 4 million
model = 'saved_models/bart-large-c4-realnewslike-yake_mask-t_2/checkpoint-689992'   # 13 million
s2t = pipeline("text2text-generation", model=model, device=device)


# 打标器，给定实体进行打标
class MyTagger:
    def __init__(self, global_mention_dict):
        all_mentions = []
        for k in global_mention_dict:
            if k != 'O':
                all_mentions += global_mention_dict[k]
        all_mentions = [tuple(m.split(' ')) for m in all_mentions]
        self.all_mentions = all_mentions
        self.mwe_tokenizer = nltk.tokenize.MWETokenizer(all_mentions,separator=' ')
        self.global_mention_dict = global_mention_dict
        self.reverse_global_mention_dict = {}
        for k in global_mention_dict:
            for e in global_mention_dict[k]:
                self.reverse_global_mention_dict[e] = k

    def tokenize(self, sentence):
        return self.mwe_tokenizer.tokenize(word_tokenize(sentence))
    
    def tag(self, sentence):
        big_words = self.tokenize(sentence) # 包含一些词组
        tags = []
        tokens = []
        for big_word in big_words:
            if big_word in self.reverse_global_mention_dict: # 该词/词组是实体
                full_tag_name = self.reverse_global_mention_dict[big_word]
                for i,single_word in enumerate(word_tokenize(big_word)):
                    if i == 0: # the first word
                        tags.append(tag_names.index('B-'+full_tag_name))
                    else: # the latter word
                        tags.append(tag_names.index('I-'+full_tag_name))
                    tokens.append(single_word)
            else:
                for single_word in word_tokenize(big_word):
                    tags.append(0)
                    tokens.append(single_word)
        assert len(tokens) == len(tags),'.'
        return tokens, tags

# 对太长的序列进行切分，可反复调用直到切不动
def cut_too_long_sequences(tokens_list, tags_list):
    short_tokens_list = []
    short_tags_list = []
    for tokens, tags in zip(tokens_list, tags_list):
        if '.' in tokens and len(tokens) > 10:
            index = tokens.index('.')
            if index < len(tokens) - 1 and index > 1:
                short_tokens_list += [tokens[:index+1], tokens[index+1:]]
                short_tags_list += [tags[:index+1], tags[index+1:]]
            else:
                short_tokens_list.append(tokens)
                short_tags_list.append(tags)
        else:
            short_tokens_list.append(tokens)
            short_tags_list.append(tags)
    assert len(short_tokens_list) == len(short_tags_list), 'not good~'
    return short_tokens_list, short_tags_list


#===============================================================================================
#                            | Augmentation Pipeline |
#===============================================================================================
num_train = 100 # 200
N_SAMPLE = 1
ADD_PROMPT = True

# 待增强的数据集：
if num_train:
    orig_dataset = raw_datasets['train'].select(range(num_train))
else:
    orig_dataset = raw_datasets['train']

# 合并短序列，方便后续抽关键词
concated_dataset = concat_multiple_sequences(orig_dataset, size=3, overlap=True)
print(f'>>> before processing, data size: {len(orig_dataset)}')
print(f'>>> after concatenation, data size: {len(concated_dataset["tokens"])}')


# 把所有实体抽出来，用于初始化tagger
global_mention_dict = {t:[] for t in list(set([t.split('-')[-1] for t in tag_names])) if t != 'O'}
for tokens, tags in zip(tqdm(concated_dataset['tokens']), concated_dataset['ner_tags']):
    mentions, mention_dict = extract_mentions(tokens, tags)
    for k in mention_dict:
        global_mention_dict[k] += mention_dict[k]
        global_mention_dict[k] += [s.lower() for s in mention_dict[k]]
        global_mention_dict[k] += [s.upper() for s in mention_dict[k]]
        global_mention_dict[k] += [s.title() for s in mention_dict[k]]
        global_mention_dict[k] += [s.capitalize() for s in mention_dict[k]]

for k in global_mention_dict:
    global_mention_dict[k] = list(set(global_mention_dict[k]))
my_tagger = MyTagger(global_mention_dict)


# 抽取实体片段
m_list = []
# topic_prompts = ['Politics: ', 'Economics: ', '']
topic_prompts = [''] # 等价于不加prompt
all_kws = []
for tokens, tags in zip(tqdm(concated_dataset['tokens']), concated_dataset['ner_tags']):
    ## 方案一：抽取mention所在的小spans，然后生成
    # spans, m = extract_mention_spans(tokens, tags)
    # m = re.sub('[.,?!:\'\"/\(\)\[\]\{\}]', '', m) # 去除一些标点
    ## 方案二：抽取关键词和mention，把其余部分都mask掉
    # t = ' '.join(tokens)
    # kws = keynotes_yake(t, max_ngram=3, topk=max(len(tokens)//8,5))
    # mentions, _ = extract_mentions(tokens, tags)
    # m = mask_unimportant_parts(t, kws+mentions)
    ## 方案三：还是统一使用semantic抽取方法，把entities当做aspect
    t = ' '.join(tokens)
    mentions, _ = extract_mentions(tokens, tags)
    kws = keysents_semantic(t, aspect_keywords=mentions, max_ngram=3, topk=max(len(tokens)//4,5))
    all_kws += [w for w in kws if not w.isupper()]
    m = mask_unimportant_parts(t, kws)
    if ADD_PROMPT:
        prompt = random.choice(topic_prompts)
        pm = prompt + m
        m_list.append(pm)
    # print(f'>>> m: {m}')

m_dataset = MDataset(m_list)

## generating
orig_augmented_dataset = defaultdict(list)
for i in range(N_SAMPLE):
    for out in tqdm(s2t(m_dataset, num_beams=3, do_sample=True, max_length=100, length_penalty=1, batch_size=32)):
        generated_text = out[0]['generated_text']
        for prompt in topic_prompts:
            generated_text = generated_text.replace(prompt,'')
        generated_text = clean_pipeline(generated_text)
        # print(f'  >>> generated_text: {generated_text}')
        ## tag the generated sentence
        new_tokens, new_tags = my_tagger.tag(generated_text)
        orig_augmented_dataset['tokens'].append(new_tokens)
        orig_augmented_dataset['ner_tags'].append(new_tags)
print(f'>>> Num of originally generated examples: {len(orig_augmented_dataset["tokens"])}')
from copy import deepcopy
augmented_dataset = deepcopy(orig_augmented_dataset)
' '.join(augmented_dataset['tokens'][4])

# 保存数据集
df = pd.DataFrame(augmented_dataset)
file_name = f'ner_data/{dataset_name}-{num_train}-S2T-aug{N_SAMPLE}_long.pkl' #_random_topic
df.to_pickle(file_name)  # 不能保存csv，因为涉及到保存list，csv会变成string了
print(file_name)


# ----------------- New Try: 随机组合keynotes来构成sketch ----------------
# N_GEN= 100
# sketches = []
# for _ in range(N_GEN):
#     candidates = random.sample(all_kws,8)
#     sketch = '<mask> ' + ' <mask> '.join(candidates) + ' <mask>'
#     sketches.append(sketch)

# free_generated_contents = []
# free_augmented_dataset = defaultdict(list)
# sketch_dataset = S2T_Dataset(sketches)
# for out in tqdm(s2t(sketch_dataset, max_length=100, batch_size=32, repetition_penalty=2.)):
#     generated_text = out[0]['generated_text']
#     generated_text = clean_pipeline(generated_text)
#     print(generated_text)
#     free_generated_contents.append(generated_text)
#     new_tokens, new_tags = my_tagger.tag(generated_text)
#     free_augmented_dataset['tokens'].append(new_tokens)
#     free_augmented_dataset['ner_tags'].append(new_tags)
    
# ' '.join(free_augmented_dataset['tokens'][0])

# # 保存数据集
# df_free = pd.DataFrame(free_augmented_dataset)
# file_name = f'ner_data/conll03-{num_train}-S2T-RC-{N_GEN}_long.pkl'
# df_free.to_pickle(file_name)  
# print(file_name)

# augmented_dataset = deepcopy(free_augmented_dataset)

# ------------------------------------------------------------------------


# Cutting: 把生成的长序列切短
print(f'cutting long seqs...')
# cutting some too-long generated sequences
augmented_dataset['tokens'], augmented_dataset['ner_tags'] = \
cut_too_long_sequences(*cut_too_long_sequences(augmented_dataset['tokens'],augmented_dataset['ner_tags']))
# do it again:
augmented_dataset['tokens'], augmented_dataset['ner_tags'] = \
cut_too_long_sequences(*cut_too_long_sequences(augmented_dataset['tokens'],augmented_dataset['ner_tags']))
print(f'>>> Num of generated training examples: {len(augmented_dataset["tokens"])}')

# Simple Filtering: 并进行一些简单过滤
print(f'filtering too short seqs...')
tokens_list = []
tags_list = []
for tokens,tags in zip(augmented_dataset['tokens'], augmented_dataset['ner_tags']):
    if len(list(set(tags))) > 1 and len(tags) > 5: # 只保留至少一个实体的,且词数大于5
        tokens_list.append(tokens)
        tags_list.append(tags)

augmented_dataset['tokens'], augmented_dataset['ner_tags'] = tokens_list, tags_list
print(f'>>> Num of generated training examples: {len(augmented_dataset["tokens"])}')


# 保存数据集
df = pd.DataFrame(augmented_dataset)
# file_name = f'ner_data/{dataset_name}-{num_train}-K2T-frame-aug{N_SAMPLE}_short_v5.pkl' # _random_topic
file_name = f'ner_data/{dataset_name}-{num_train}-S2T-RC-{200}_short.pkl'
df.to_pickle(file_name)  # 不能保存csv，因为涉及到保存list，csv会变成string了
print(file_name)




# # ## 如果想方便查看增强的样本，可以通过下面方式生成txt文档：
# from datasets import Dataset as HFDataset
# import pandas as pd
# f_name = f'ner_data/conll03-{num_train}-K2T-frame-aug{N_SAMPLE}_orig_v2'
# df = pd.read_pickle('%s.pkl'%f_name)
# augmented_dataset = HFDataset.from_pandas(df)

# with open('%s.txt'%f_name,'w') as f:
#     f.write('')
# with open('%s.txt'%f_name,'a') as f:
#     for i in range(len(augmented_dataset['tokens'])):
#         show_dataset(augmented_dataset,i,print_to_file=f)


