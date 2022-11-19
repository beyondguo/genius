import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from collections import defaultdict
from datasets import load_dataset
nltk.download('stopwords')
nltk.download('punkt')
import random
random.seed(5)
import sys
sys.path.append('..')
from genius_utils import SketchExtractor, clean_pipeline

# load raw dataset
# c4-realnewslike, about 14 million documents
import os
os.system('export HF_DATASETS_CACHE="../saved_datasets/hf_cache"')
os.system('export HF_DATASETS_OFFLINE=1')
raw_dataset = load_dataset('c4','realnewslike', cache_dir='../saved_datasets/hf_cache')
print(raw_dataset)



sketch_extractor = SketchExtractor(model='yake')

def a_len(s):# approximate length
    return len(s.split(' ')) 


def text_preprocess(examples):
    res = defaultdict(list)
    documents = examples['text']
    # the following processing is mainly for C4 corpus：
    for document in documents:
        document = clean_pipeline(document)
        document = document.replace('\n',' ').replace('  ',' ')
        document = document.replace('\'s','')
        sents = sent_tokenize(document)
        res['passage'].append(' '.join(sents[:15]))
        # random_sent = random.choice(sents)
        # res['sentence'].append(random_sent)
    return res

preporcessed_dataset = raw_dataset.map(text_preprocess,batched=True,\
                            remove_columns=raw_dataset['train'].column_names,\
                            batch_size=100, num_proc=50) 
print(preporcessed_dataset)

def add_sketch_to_dataset(examples):
    res = defaultdict(list)
    passages = examples['passage']
    # sents = examples['sentence']

    # for p,s in zip(passages,sents):
    #     # passage:
    #     res['text'].append(p)
    #     _, kws = sketch_extractor.get_kws(p, max_ngram=3, top=max(a_len(p)//5,1)) # max 3-gram
    #     for i in [1,2,3,4]:
    #         sketch = sketch_extractor.get_sketch_from_kws(p, kws, template=i)
    #         res['sketch_%s'%i].append(sketch)
    #     # sentence:
    #     res['text'].append(s)
    #     _, kws = sketch_extractor.get_kws(s, max_ngram=2, top=max(a_len(s)//5,1)) # max 2-gram
    #     for i in [1,2,3,4]:
    #         sketch = sketch_extractor.get_sketch_from_kws(s, kws, template=i)
    #         res['sketch_%s'%i].append(sketch)

    for p in passages:
        # passage:
        res['text'].append(p)
        _, kws = sketch_extractor.get_kws(p, max_ngram=3, top=max(a_len(p)//5,1)) # max 3-gram
        sketch = sketch_extractor.get_sketch_from_kws(p, kws, template=4)
        res['sketch_4'].append(sketch)
    return res



dataset_with_sketch = preporcessed_dataset.map(add_sketch_to_dataset, batched=True, 
                                                remove_columns=preporcessed_dataset['train'].column_names,
                                                batch_size=10, num_proc=200)
print(dataset_with_sketch)


name = "c4-realnewslike-passage-max15sents"
dataset_with_sketch.save_to_disk(f'../saved_datasets/{name}_{len(dataset_with_sketch)}/')

