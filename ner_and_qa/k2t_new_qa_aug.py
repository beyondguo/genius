import argparse
from transformers import pipeline
import types
import random
random.seed(5)
from datasets import load_dataset

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--n_train', type=int, help='original train size')
parser.add_argument('--n_aug', type=int, default=1, help='num of augmentation times')
parser.add_argument('--device', type=int, default=7, help='the gpu device No.')
args = parser.parse_args()


raw_datasets = load_dataset("squad")


# sega生成器
device = int(args.device)
# model = f'saved_models/bart-large-c4-l_50_200-d_13799838-yake_mask-t_3900800/checkpoint-152375'
model = 'saved_models/bart-large-c4-realnewslike-yake_mask-t_2/checkpoint-689992'  # 13 million
sega = pipeline("text2text-generation", model=model, device=device)


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

from aspect_keybert import AspectKeyBERT
akb = AspectKeyBERT()

def keynotes_semantic(text, aspect_keywords=None, max_ngram=3, topk=20, aspect_only=False):
    if type(topk) == types.FunctionType: # 使用一个函数来确定topk
        topk_num = topk(text)
    else:
        topk_num = topk
    kws_paris = akb.extract_aspect_keywords(text,top_n=topk_num, keyphrase_ngram_range=(1,max_ngram),
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
    words = word_tokenize(text)
    for w in words:
        if "n't" in w or w in ['not','but','no','nothing','however','though','nevertheness']:
            kws.append(w)
    words_idxs = []
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



N_TRAIN = args.n_train
N_AUG = args.n_aug
# N_TRAIN = 200
# N_AUG = 3
print(f'>>>>>> N_TRAIN: {N_TRAIN}')
print(f'>>>>>> N_AUG: {N_AUG}')


import nltk
nltk.download('punkt')
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")
# 待增强的数据集（一般都从train里面取子集）
if N_TRAIN:
    orig_datasets = raw_datasets['train'].select(range(N_TRAIN))
else:
    orig_datasets = raw_datasets['train']
contexts = orig_datasets['context']
questions = orig_datasets['question']
answer_dicts = orig_datasets['answers'] # 发现有很多很短的答案


def get_topk(s,max_k=8):
    return max(min(len(s)//10+1, max_k),5)

# constructing masked samples:
print('** Constructing masked samples......')
m_contexts = []
m_questions = []
m_answers = []
m_answer_sents= []
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
        # 使用wanted_sentences中的首个句子
        start = context.index(wanted_sentences[0].strip())
        end = start + len(wanted_sentences[0])
        
        # find and mask pre-context
        pre_context = context[:start]
        if pre_context != '': 
            kws = keynotes_semantic(pre_context, aspect_keywords=[question], topk=get_topk(pre_context))
            m_pre_context = mask_unimportant_parts(pre_context, kws)
        else: # 没有上文，补一个mask
            m_pre_context = '<mask> '

        # find and mask post-context
        post_context = context[end:]
        if post_context != '': 
            kws = keynotes_semantic(post_context, aspect_keywords=[question], topk=get_topk(post_context))
            m_post_context = mask_unimportant_parts(post_context, kws)
        else: # 没有下文，补一个mask
            m_post_context = ' <mask>'

        # concatenate into a new context, and determine the new answer start
        answer_sent = wanted_sentences[0]
        new_context = m_pre_context + answer_sent + m_post_context

        m_contexts.append(new_context)
        m_questions.append(question)
        m_answer_sents.append(answer_sent)
        m_answers.append(answer)
    
    except Exception as e:
        print(e)
        # for debug:
        # print(f'>>> C:{context}')
        # print(f'>>> Q:{question}')
        # print(f'>>> A:{answer}')
        # print(f'>>> more: {answer_more}')


print('** Working Hard to Augment Your Dataset......')
m_dataset = MDataset(m_contexts)
generated_contexts = []
for _ in range(N_AUG): # 增强多次
    for out in tqdm(sega(m_dataset, num_beams=3, do_sample=True, max_length=200, length_penalty=2, batch_size=32,repetition_penalty=2.)): # 原来200, no repetition_penalty
        generated_text = out[0]['generated_text']
        generated_contexts.append(generated_text)

augmented_dataset = defaultdict(list)
mismatched = []
for i, c, q, a_s, a in zip(range(len(m_answers*N_AUG)),generated_contexts, m_questions*N_AUG,m_answer_sents*N_AUG, m_answers*N_AUG):
    a_s_idx = -1
    try:
        a_s_idx = c.index(a_s) # index of the answer sentence
    except Exception as e:
        # 一个严重的问题，原始的句子不一定会原封不动地输出，可能会有些微小变化
        # 这样原来的answer sent就不一定找得到了，最好能用近似匹配，即重合率高于某阈值即可
        sents = sent_tokenize(c)
        for s in sents:
            words = word_tokenize(s)
            orig_words = word_tokenize(a_s)
            n = len([w for w in words if w in orig_words])
            # 重合率达到0.6，且answer也在该句子中，说明这个句子就对应原始答案句
            if n/len(words) > 0.6 and a in s: 
                a_s = s
                a_s_idx = c.index(a_s)
                break
    if a_s_idx > -1: # 确认找到了答案句子
        start = a_s_idx + a_s.index(a)
        assert c[start:start+len(a)] == a, '%s Answer Position Mismatch!'%i

        # add to new dataset
        augmented_dataset['context'].append(c)
        augmented_dataset['question'].append(q)
        augmented_dataset['answers'].append({'text': [a], 'answer_start': [start]})
    else:
        mismatched.append(i)
print('mismatch rate: ',len(mismatched)/len(generated_contexts))  

# double check:
print(len(augmented_dataset['context']))
for c,q,a_d in zip(augmented_dataset['context'],augmented_dataset['question'],augmented_dataset['answers']):
#     print(f'>>>{c}')
#     print(f'>>>{q}')
#     print(f'>>>{a_d["text"]}')
    answer = a_d['text'][0]
    start = a_d['answer_start'][0]
    assert c[start:start+len(answer)] == answer, 'Not Good!'

print(len(augmented_dataset['context']))

import pandas as pd
df = pd.DataFrame(augmented_dataset)
f_name = f'qa_data/squad_first{N_TRAIN}_aug{N_AUG}_S2T.pkl'
df.to_pickle(f_name)
print(f_name)