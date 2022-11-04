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
from sega_utils import SketchExtractor, clean_pipeline

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
                            batch_size=100, num_proc=50) # 7-8 minutes
print(preporcessed_dataset)

def add_sketch_to_dataset(examples):
    """
    先直接用\n\n给分成很多段落，然后把太短的过滤掉
    接着抽取关键词(总词数的的1/10)
    由于样本增多了，记得在map的时候设置remove_columns=your_dataset['train'].column_names
    """
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
        res['sketch_%s'%4].append(sketch)
    return res



"""
example_contents = preporcessed_dataset['validation'][10:20]
processed = add_sketch_to_dataset(example_contents)

example_contents = raw_dataset['validation']['text'][10:20]
processed = add_kws_to_dataset_longtext({'text':example_contents})
i = 10
processed['text'][i]
processed['sketch_1'][i]
processed['sketch_2'][i]
processed['sketch_3'][i]
processed['sketch_4'][i]
"""

dataset_with_sketch = preporcessed_dataset.map(add_sketch_to_dataset, batched=True, 
                                                remove_columns=preporcessed_dataset['train'].column_names,
                                                batch_size=10, num_proc=200)

# 如果只训练一个子集：
# num_doc = 13799838
# # C4 train中总共文档数 13799838
# dataset_with_kws = dataset4['train'].select(random.sample(range(13799838),num_doc))\
#                                 .map(add_kws_to_dataset_longtext,batched=True,\
#                                  remove_columns=dataset4['train'].column_names,\
#                                  batch_size=100,num_proc=500,)nvidia-smi


# 全量数据集：
# dataset_with_kws = raw_dataset.map(add_kws_to_dataset_longtext,batched=True,\
#                             remove_columns=raw_dataset['train'].column_names,\
#                             batch_size=10, num_proc=200) # ,

print(dataset_with_sketch)

# def show(i):
#     print(f">>>text:\n{dataset_with_kws['text'][i]}")
#     print(f">>>kws:\n{dataset_with_kws['keynote'][i]}")


name = "c4-realnewslike-4templates-passage-max15sents"
dataset_with_sketch.save_to_disk(f'../saved_datasets/{name}_{len(dataset_with_sketch)}/')


#-------------------
# import os
# os.system("sh oc.sh")
