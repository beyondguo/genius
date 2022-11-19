"""
todo:
"""
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
from rake_nltk import Rake
from keybert import KeyBERT
import yake
from collections import defaultdict
from datasets import load_dataset, Dataset
import random
random.seed(5)

raw_datasets = load_dataset("conll2003")



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
def concat_multiple_sequences(dataset, size=3):
    # 传入正经的huggingface dataset格式
    # 如果是子集的话，建议使用select方法来筛选
    new_dataset = defaultdict(list)
    l = len(dataset)
    # n = l // size
    for i in range(l-size):
        concat_tokens = np.concatenate(dataset[i:i+size]['tokens'])
        concat_tags = np.concatenate(dataset[i:i+size]['ner_tags'])
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
    


# 关键词抽取工具
max_ngrams = 1
# keywords extraction:
ke_rake = Rake()  # .extract_keywords_from_text(text).get_ranked_phrases()
ke_yake = yake.KeywordExtractor(n=max_ngrams,top=10) # n for max_ngram, use .extract_keywords(text)
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
        keyphrase_ngram_range=(1, max_ngrams),top_n=topk)
        kws =  [pair[0] for pair in kws_paris]
    return kws


# 结合mention和keywords
def extract_mentions_with_keywords(tokens, tags, n=1, convert_to_str=False):
    mentions, mention_dict = extract_mentions(tokens, tags)
    text = ' '.join(tokens)
    keywords = extract_keywords(text, 'keybert', topk=15)
    # 然后需要把keywords中已经出现在mention中的词，尤其是是对词组的部分的词，要去掉
    # 由于keybert等工具抽取关键词时，会自动转换成小写（这太傻比了），这导致很多地名人名变成小写，因此下面还需要都转成lower去识别
    keywords = [w for w in keywords if w.lower() not in ' '.join(mentions).lower()]

    res = []
    for i in range(n):
        words = random.choices(mentions,k=min(2,len(mentions))) + random.sample(keywords,min(10,len(keywords))) # 所有mentions以及随机一些keywords
        if convert_to_str:
            res.append(' '.join(random.sample(words, len(words))))
        else:
            res.append(random.sample(words, len(words)))
    return res



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
    # 当超过三个句子，且最后一个句子不以标点结尾，就移除最后一句
    sents = sent_tokenize(text)
    text = ' '.join(sents)
    if len(sents) > 3:
        if sents[-1][-1] not in ".?!。？！\'\"":
            text = ' '.join(sents[:-1])
    return text
        
def clean_pipeline(text):
    return ((remove_special_characters(remove_brakets(text))))



# K2T生成器
class K2T_Generator:
    def __init__(self,model, device):
        self.generator = pipeline("text2text-generation", model=model, device=device)
    def k2t(self, inputs, beam=1, sample=False, max_length=200):
        """
        inputs: text or a list of text
        """
        res = self.generator(inputs, num_beams=beam, do_sample=sample, num_return_sequences=1, max_length=max_length)
        return [item['generated_text'] for item in res]


model = f'saved_models/bart-large-cnn-wikipedia-paras-yake-importance-10000d-final'
generator = K2T_Generator(model, device=1)


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
num_train = 500
num_aug = 1


# 待增强的数据集：
orig_dataset = raw_datasets['train'].select(range(num_train))
# 合并短序列，方便后续抽关键词
concated_dataset = concat_multiple_sequences(orig_dataset, size=3)
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


# 开始逐个抽取关键信息、生成、打标
augmented_dataset = defaultdict(list)
for tokens, tags in zip(tqdm(concated_dataset['tokens']), concated_dataset['ner_tags']):
    ## key phrases extracting
    extracted_phrases_text_list = extract_mentions_with_keywords(tokens, tags, n=num_aug, convert_to_str=True)
    ## k2t generating
    generated_text_list = generator.k2t(extracted_phrases_text_list)
    print(f'>>> extracted_phrases_text: {extracted_phrases_text_list}')
    for generated_text in generated_text_list:
        generated_text = clean_pipeline(generated_text)
        print(f'  >>> generated_text: {generated_text}')

        ## tag the generated sentence
        new_tokens, new_tags = my_tagger.tag(generated_text)
        augmented_dataset['tokens'].append(new_tokens)
        augmented_dataset['ner_tags'].append(new_tags)

print(f'>>> Num of originally generated examples: {len(augmented_dataset["tokens"])}')

# 把生成的长序列切短，并进行一些简单过滤
print(f'cutting long seqs...')
# cutting some too-long generated sequences
augmented_dataset['tokens'], augmented_dataset['ner_tags'] = \
cut_too_long_sequences(*cut_too_long_sequences(augmented_dataset['tokens'],augmented_dataset['ner_tags']))
# do it again:
augmented_dataset['tokens'], augmented_dataset['ner_tags'] = \
cut_too_long_sequences(*cut_too_long_sequences(augmented_dataset['tokens'],augmented_dataset['ner_tags']))
print(f'>>> Num of generated training examples: {len(augmented_dataset["tokens"])}')

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
df.to_pickle(f'ner_data/conll03-{num_train}-k2t-free-aug{num_aug}.pkl')  # 不能保存csv，因为涉及到保存list，csv会变成string了



# ## 如果想方便查看增强的样本，可以通过下面方式生成txt文档：
# from datasets import Dataset
# import pandas as pd
# df = pd.read_pickle(f'ner_data/conll03-{num_train}-newK2T-aug{num_aug}.pkl')
# augmented_dataset = Dataset.from_pandas(df)

# with open('ner_data/conll03-{num_train}-newK2T-aug{num_aug}.txt','a') as f:
#     for i in range(len(augmented_dataset['tokens'])):
#         show_dataset(augmented_dataset,i,print_to_file=f)