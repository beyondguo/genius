import argparse

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--n_train', type=int, help='original train size')
parser.add_argument('--n_aug', type=int, default=1, help='num of augmentation times')
parser.add_argument('--device', type=int, default=0, help='the gpu device No.')
args = parser.parse_args()

from transformers import pipeline
class K2T_Generator:
    def __init__(self,model, device):
        self.generator = pipeline("text2text-generation", model=model, device=device)
    def k2t(self, inputs, beam=1, sample=False, max_length=200):
        """
        inputs: text or a list of text
        """
        res = self.generator(inputs, num_beams=beam, do_sample=sample, num_return_sequences=1, max_length=max_length)
        return [item['generated_text'] for item in res]


## 10000 documents for pre-training
model_path = f'saved_models/bart-large-cnn-wikipedia-paras-yake-importance-10000d-final'
## one million wikipedia documents for pretraining
# model_path = f'saved_models/bart-large-cnn-wikipedia-paras-yake-importance-1000000d-final'
generator = K2T_Generator(model_path, device=int(args.device))

import random
random.seed(5)
from aspect_keybert import AspectKeyBERT
akb = AspectKeyBERT()
def extract_aspect_keywords(text, aspect_keywords, aspect_as_doc=False, topk=10, n_gram=2, 
                            add_aspect_keywords=0, shuffle=False, return_str=True):
    """
    aspect_keywords: list, 用于计算aspect embedding
    aspect_as_doc: bool, 若为True，则将aspect作为抽取关键词的完全参照
    add_aspect_keywords: int, 将n个aspect words插入到抽取的关键词中一并返回，0则代表不插入
    
    return_str: 是否拼接成一个字符串返回
    """    
    kws_paris = akb.extract_aspect_keywords(text, use_aspect_as_doc_embedding=aspect_as_doc, top_n=topk,\
                                            keyphrase_ngram_range=(1,n_gram),aspect_keywords=aspect_keywords)
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


from datasets import load_dataset
raw_datasets = load_dataset("squad")


import re
def get_that_sentence(text, that):
    # 保证及时在开头结尾也能识别
    text = '.' + text + '.'
    # 当要匹配的东西里面包含了括号
    that = that.replace('(','\(')
    that = that.replace(')','\)')
    that = that.replace('([','\[')
    that = that.replace(']','\]')
    that = that.replace('$','\$')
    that = that.replace('.','\.')
    return re.findall(r'[\.?!]([^\.?!]*?%s[^\.?!]*?)[\.?!]'%that, text)
# word = 'heihei.'
# t = 'what a nice! day!. what? heihei.'
# get_that_sentence(t,word)





N_AUG = args.n_aug
N_TRAIN = args.n_train
def get_topk(s,max_k=8):
    return max(min(len(s)//10+1, max_k),5)

import nltk
nltk.download('punkt')
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")
# 待增强的数据集（一般都从train里面取子集）
orig_datasets = raw_datasets['train'].select(range(N_TRAIN))
contexts = orig_datasets['context']
questions = orig_datasets['question']
answer_dicts = orig_datasets['answers'] # 发现有很多很短的答案

# augmentation loop
"""TO DO: """
augmented_dataset = defaultdict(list)
print('** Working Hard to Augment Your Dataset......')
for context, question, answer_dict in zip(tqdm(contexts), questions, answer_dicts):
    # 答案文本可能出现在多个句子中，但并不是每个句子都是能回答问题的句子
    # 如果需要改进的话，最好根据相似度做一个排序
    # 一个简易版本就是，直接吧所以这些句子都拼起来，都保留下来
    # 上下文就都是用第一个句子的上下文
    answer = answer_dict['text'][0]
    init_start = answer_dict['answer_start'][0]
    init_end = init_start+len(answer)
    
    # locate the sentence where answer appear
    # 为了找到唯一的答案句子，把答案文本前后扩展一点点
    answer_more = context[max(0, init_start-2) : min(len(context), init_end+2)]
    
    
    try:
        wanted_sentences = get_that_sentence(context, answer_more)
        if len(wanted_sentences) == 0:
            print('zero wanted_sentences')
            print(f'>>> C:{context}')
            print(f'>>> Q:{question}')
            print(f'>>> A:{answer}')
            print(f'>>> more: {answer_more}')
            wanted_sentences = []
            sents = sent_tokenize(context)
            for s in sents:
                if answer in s:
                    wanted_sentences.append(s)
            print(f'>>>S: {wanted_sentences}')
            if len(wanted_sentences) == 0:
                continue # 还没有，直接跳过吧
        elif len(wanted_sentences) > 1:
            print('too many wanted_sentences')
            print(f'>>> C:{context}')
            print(f'>>> Q:{question}')
            print(f'>>> A:{answer}')
            print(f'>>> more: {answer_more}')
            print(f'>>> {wanted_sentences}')
        start = context.index(wanted_sentences[0].strip())
        end = start + len(wanted_sentences[0])
            
        # 可增强多次，记得开启shuffle
        for _ in range(N_AUG):
            # generate pre-context
            pre_context = context[:start]
            if pre_context != '': # 没有上文
                pre_kws = extract_aspect_keywords(pre_context, [question], topk=get_topk(pre_context), return_str=True, shuffle=True)
                generated_pre_context = generator.k2t(pre_kws)[0] + ' '
            else: # 没有上文，就直接为空
                generated_pre_context = ''

            # generate post-context
            post_context = context[end:]
            if post_context != '': # 没有下文
                post_kws = extract_aspect_keywords(post_context, [question], topk=get_topk(post_context), return_str=True, shuffle=True)
                generated_post_context = ' ' + generator.k2t(post_kws)[0]
            else:
                generated_post_context = ''

            # concatenate into a new context, and determine the new answer start
            answer_span = ' '.join(wanted_sentences)
            new_context = generated_pre_context + answer_span + generated_post_context
            new_start = len(generated_pre_context)+answer_span.index(answer)
            assert new_context[new_start:new_start+len(answer)] == answer, 'Answer Position Mismatch!'

            # add to new dataset
            augmented_dataset['context'].append(new_context)
            augmented_dataset['question'].append(question)
            augmented_dataset['answers'].append({'text': [answer], 'answer_start': [new_start]})
    except Exception as e:
        print(e)
        # for debug:
        print(f'>>> C:{context}')
        print(f'>>> Q:{question}')
        print(f'>>> A:{answer}')
        print(f'>>> more: {answer_more}')


# double check:
print(len(augmented_dataset['context']))
for c,q,a_d in zip(augmented_dataset['context'],augmented_dataset['question'],augmented_dataset['answers']):
#     print(f'>>>{c}')
#     print(f'>>>{q}')
#     print(f'>>>{a_d["text"]}')
    answer = a_d['text'][0]
    start = a_d['answer_start'][0]
    assert c[start:start+len(answer)] == answer, 'Not Good!'


import pandas as pd
df = pd.DataFrame(augmented_dataset)
df.to_pickle(f'qa_data/squad_first{N_TRAIN}_aug{N_AUG}_v3.pkl')