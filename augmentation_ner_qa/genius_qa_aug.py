import re
import sys
import argparse
from transformers import pipeline
import types
import random
random.seed(5)
from datasets import load_dataset
sys.path.append('../')
# from aspect_keybert import AspectKeyBERT
from genius_utils import SketchExtractor, List2Dataset
sketch_extractor = SketchExtractor('bert')
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--n_train', type=int, default=50, help='original train size')
parser.add_argument('--n_aug', type=int, default=1, help='num of augmentation times')
parser.add_argument('--device', type=int, default=7, help='the gpu device No.')
args = parser.parse_args()

raw_datasets = load_dataset("squad")

# genius model
device = int(args.device)
model = 'beyond/genius-large'
genius = pipeline("text2text-generation", model=model, device=device)


def get_that_sentence(text, that):
    text = '.' + text + '.' # to guatentee the recognition even at beginning/end
    that = that.replace('(','\(')
    that = that.replace(')','\)')
    that = that.replace('([','\[')
    that = that.replace(']','\]')
    that = that.replace('$','\$')
    that = that.replace('.','\.')
    return re.findall(r'[\.?!]([^\.?!]*?%s[^\.?!]*?)[\.?!]'%that, text)



N_TRAIN = args.n_train
N_AUG = args.n_aug
print(f'>>>>>> N_TRAIN: {N_TRAIN}')
print(f'>>>>>> N_AUG: {N_AUG}')


import nltk
nltk.download('punkt')
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

if N_TRAIN:
    orig_datasets = raw_datasets['train'].select(range(N_TRAIN))
else:
    orig_datasets = raw_datasets['train']
contexts = orig_datasets['context']
questions = orig_datasets['question']
answer_dicts = orig_datasets['answers'] 


def get_topk(s,max_k=8):
    return max(min(len(s)//10+1, max_k),5)

# constructing masked samples:
print('** Constructing masked samples......')
m_contexts = []
m_questions = []
m_answers = []
m_answer_sents= []
for context, question, answer_dict in zip(tqdm(contexts), questions, answer_dicts):
    # SQuAD has lots of short answers, 
    # Therefore, the answer string may appear in more than one sentence, but not every sentence is correct.
    # currently we use a straghtforway to handle this, by concatenating all the possible sentences
    # and use the contexts for the first sentence as the contexts
    answer = answer_dict['text'][0]
    init_start = answer_dict['answer_start'][0]
    init_end = init_start+len(answer)
    
    # locate the sentence where answer appear
    # we expand the answer string a bit longer to match more precisely
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
                continue 
        elif len(wanted_sentences) > 1:
            print('too many wanted_sentences')
            print(f'>>> C:{context}')
            print(f'>>> Q:{question}')
            print(f'>>> A:{answer}')
            print(f'>>> more: {answer_more}')
            print(f'>>> {wanted_sentences}')
        
        start = context.index(wanted_sentences[0].strip())
        end = start + len(wanted_sentences[0])
        
        # find and mask pre-context
        pre_context = context[:start]
        if pre_context != '': 
            # kws = keynotes_semantic(pre_context, aspect_keywords=[question], topk=get_topk(pre_context))
            # m_pre_context = mask_unimportant_parts(pre_context, kws)
            _, kws = sketch_extractor.get_kws(pre_context, aspect_keywords=[question], top=get_topk(pre_context))
            m_pre_context = sketch_extractor.get_sketch_from_kws(pre_context, kws)
        else: # no pre-context, add a mask
            m_pre_context = '<mask> '

        # find and mask post-context
        post_context = context[end:]
        if post_context != '': 
            # kws = keynotes_semantic(post_context, aspect_keywords=[question], topk=get_topk(post_context))
            # m_post_context = mask_unimportant_parts(post_context, kws)
            _, kws = sketch_extractor.get_kws(post_context, aspect_keywords=[question], top=get_topk(post_context))
            m_post_context = sketch_extractor.get_sketch_from_kws(post_context, kws)
        else: # no post-context, add a mask
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
m_dataset = List2Dataset(m_contexts)
generated_contexts = []
for _ in range(N_AUG): 
    for out in tqdm(genius(m_dataset, num_beams=3, do_sample=True, max_length=200, length_penalty=2, batch_size=32,repetition_penalty=2.)): 
        generated_text = out[0]['generated_text']
        generated_contexts.append(generated_text)

augmented_dataset = defaultdict(list)
mismatched = []
for i, c, q, a_s, a in zip(range(len(m_answers*N_AUG)),generated_contexts, m_questions*N_AUG,m_answer_sents*N_AUG, m_answers*N_AUG):
    a_s_idx = -1
    try:
        a_s_idx = c.index(a_s) # index of the answer sentence
    except Exception as e:
        # a problem using GENIUS is that the original input sentence may have small changes,
        # resulting in the mismatch in output sequence
        # therefore we calculate an overlap ratio to find the right sentence
        sents = sent_tokenize(c)
        for s in sents:
            words = word_tokenize(s)
            orig_words = word_tokenize(a_s)
            n = len([w for w in words if w in orig_words])
            # overlap > 0.6 and the answer is also in the sentence, then this is the right sentence we want
            if n/len(words) > 0.6 and a in s: 
                a_s = s
                a_s_idx = c.index(a_s)
                break
    if a_s_idx > -1: # we've got the right answer
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
f_name = f'../qa_data/squad_first{N_TRAIN}_aug{N_AUG}_genius.pkl'
df.to_pickle(f_name)
print(f_name)