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
from genius_utils import SketchExtractor
import os
from tqdm import tqdm
import jieba
from datasets import Dataset
sketch_extractor = SketchExtractor(model='jieba')

# paragraphs = []
# files = os.listdir('../../zh_clue/')

# for file in tqdm(files):
#     if '.txt' not in file:
#         continue
#     with open('../../zh_clue/'+file,'r',encoding='utf8') as f:
#         lines = f.readlines()
#         lines = [remove_brakets(l) for l in lines if len(l)>50 and len(l)<300]
#         paragraphs += lines
# print('>>> num of paragraphs:', len(paragraphs))

# import re
# def contain_nonchinese(s):
#     if re.findall('[^\u4e00-\u9fa5，。、！？： ]+', s):
#         return True
#     return False

# clean_paragraphs = [p.replace('\n','') for p in paragraphs]  # 331897788
# onlyzh_paragraphs = [p for p in clean_paragraphs if not contain_nonchinese(p)]  # 88328203

# extract sketches
# sketches = []
# for p in tqdm(onlyzh_paragraphs):
#     _, kws = sketch_extractor.get_kws(p, top=max(len(jieba.lcut(p))//5,1))
#     sketch = sketch_extractor.get_sketch_from_kws(p, kws, mask='[MASK]',sep='')
#     sketches.append(sketch)

# passage_dataset = Dataset.from_dict({'passage':onlyzh_paragraphs})
# passage_dataset.push_to_hub("beyond/chinese_clean_passages_80m")
# passage_dataset.save_to_disk(f'../saved_datasets/chinese_clean_passages_80million')

passage_dataset = load_dataset('beyond/chinese_clean_passages_80m')
def add_sketch_to_dataset(examples):
    """
    """
    res = defaultdict(list)
    passages = examples['passage']

    for p in passages:
        # passage:
        res['text'].append(p)
        _, kws = sketch_extractor.get_kws(p, top=max(len(jieba.lcut(p))//5,1))
        # we plan to use `fnlp/bart-large-chinese` for pre-training, the mask token is `[MASK]`
        sketch = sketch_extractor.get_sketch_from_kws(p, kws, mask='[MASK]',sep='')
        res['sketch'].append(sketch)
    return res


dataset_with_sketch = passage_dataset.map(add_sketch_to_dataset, batched=True, 
                                            batch_size=10, num_proc=20)


print(dataset_with_sketch)

dataset_with_sketch.save_to_disk(f'../saved_datasets/chinese_clean_passages_80m_with_sketch')
