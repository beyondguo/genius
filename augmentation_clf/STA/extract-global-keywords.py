"""
加载分类数据集，把文本和label都变成list的格式
然后直接调用KeywordsExtractor的global_role_kws_extraction_one_line函数
就会把ls、lr以及各种roles关键词进行抽取并保存
"""
import pandas as pd
from keywords_extractor import KeywordsExtractor
import argparse
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--dataset_name', type=str, help='dataset dir name, in data/')
parser.add_argument('--lang', type=str, default='en', help='en or zh')
args = parser.parse_args()

# 把content和label都包装成list的形式
dataset_name = args.dataset_name
lang = args.lang
dataset = pd.read_csv(f'data/{dataset_name}/train.csv')
 # 处理空值问题
dataset = dataset.dropna()
dataset = dataset[dataset.content != '']
contents = list(dataset['content'])
label_names = list(dataset['label'])

# 如果有标签描述文件，就加载
import json
import os
label_desc_file = f'data/{dataset_name}/label_desc.json'
if os.path.exists(label_desc_file):
    print("Oh good! We have label descriptions!")
    with open(label_desc_file) as f:
        label_desc_dict = json.load(f)
else:
    label_desc_dict = None

ke = KeywordsExtractor(lang)
kws_dict = ke.global_role_kws_extraction_one_line(
                contents,label_names, 
                label_desc_dict=label_desc_dict,
                output_dir='saved_keywords',
                name=dataset_name,
                overwrite=True)
