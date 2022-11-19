##################
# back-translation 
##################
import argparse
from transformers import pipeline
import types
import random
random.seed(5)
from datasets import load_dataset

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--n_train', type=int, help='original train size')
parser.add_argument('--n_aug', type=int, default=1, help='num of augmentation times')

args = parser.parse_args()


raw_datasets = load_dataset("squad")


# translator
# device = int(args.device)
model1 = 'Helsinki-NLP/opus-mt-en-de'
model2 = 'Helsinki-NLP/opus-mt-de-en'
translator1 = pipeline("translation", model=model1, device=2)
translator2 = pipeline("translation", model=model2, device=3)


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



from torch.utils.data import Dataset
class MyDataset(Dataset):
    def __init__(self, contents):
        self.contents = contents
    def __len__(self):
        return len(self.contents)
    def __getitem__(self, i):
        return self.contents[i]



N_TRAIN = args.n_train
N_AUG = args.n_aug
# N_TRAIN = 50
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

# 收集上下文
print('** Constructing masked samples......')
m_pre_contexts = []
m_post_contexts = []
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
            wanted_sentences = []
            sents = sent_tokenize(context)
            for s in sents:
                if answer in s:
                    wanted_sentences.append(s)
            if len(wanted_sentences) == 0:
                continue # 还他妈没有，直接跳过吧
        elif len(wanted_sentences) > 1:
            pass
        # 使用wanted_sentences中的首个句子
        start = context.index(wanted_sentences[0].strip())
        end = start + len(wanted_sentences[0])
        
        # find the pre- and post-context
        pre_context = context[:start]
        post_context = context[end:]

        # concatenate into a new context, and determine the new answer start
        answer_sent = wanted_sentences[0]

        m_pre_contexts.append(pre_context)
        m_post_contexts.append(post_context)
        m_questions.append(question)
        m_answer_sents.append(answer_sent)
        m_answers.append(answer)
    
    except Exception as e:
        print(e)



pre_dataset = MyDataset(m_pre_contexts)
inter_pre_contexts = []
print('** Translating pre-contexts to German ......')
for _ in range(N_AUG):
    for out in tqdm(translator1(pre_dataset, num_beams=3, do_sample=True, truncation=True, max_length=400, batch_size=32)):
        generated_text = out[0]['translation_text']
        inter_pre_contexts.append(generated_text)
inter_pre_dataset = MyDataset(inter_pre_contexts)
generated_pre_contexts = []
print('** Translating pre-contexts back to English ......')
for out in tqdm(translator2(inter_pre_dataset, num_beams=3, do_sample=True, truncation=True, max_length=400, batch_size=32)):
    generated_text = out[0]['translation_text']
    generated_pre_contexts.append(generated_text)



post_dataset = MyDataset(m_post_contexts)
inter_post_contexts = []
print('** Translating post-contexts to German ......')
for _ in range(N_AUG):
    for out in tqdm(translator1(post_dataset, num_beams=3, do_sample=True, truncation=True, max_length=400, batch_size=32)):
        generated_text = out[0]['translation_text']
        inter_post_contexts.append(generated_text)
inter_post_dataset = MyDataset(inter_post_contexts)
generated_post_contexts = []
print('** Translating post-contexts back to English ......')
for out in tqdm(translator2(inter_post_dataset, num_beams=3, do_sample=True, truncation=True, max_length=400, batch_size=32)):
    generated_text = out[0]['translation_text']
    generated_post_contexts.append(generated_text)


augmented_dataset = defaultdict(list)
mismatched = []
for i, c_pre, c_post, q, a_s, a in zip(range(len(m_answers*N_AUG)),generated_pre_contexts,generated_post_contexts, m_questions*N_AUG,m_answer_sents*N_AUG, m_answers*N_AUG):
    a_s_idx = -1
    c = ' '.join([c_pre, a_s, c_post])
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
print('mismatch rate: ',len(mismatched)/len(generated_pre_contexts))  


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
f_name = f'qa_data/squad_first{N_TRAIN}_aug{N_AUG}_BT.pkl'
df.to_pickle(f_name)
print(f_name)