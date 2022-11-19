
from STA.keywords_extractor import KeywordsExtractor
import os
import pandas as pd
import argparse
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--dataset_name', type=str, help='dataset dir name, in data/')
parser.add_argument('--lang', type=str, default='en', help='en or zh')
args = parser.parse_args()

# 把content和label都包装成list的形式
dataset = pd.read_csv(f'../data_clf/{args.dataset_name}/train.csv')
 # 处理空值问题
dataset = dataset.dropna()
dataset = dataset[dataset.content != '']
contents = list(dataset['content'])
label_names = list(dataset['label'])

# 如果有标签描述文件，就加载
# import json
# import os
# label_desc_file = f'../data_clf/{args.dataset_name}/label_desc.json'
# if os.path.exists(label_desc_file):
#     print("Oh good! We have label descriptions!")
#     with open(label_desc_file) as f:
#         label_desc_dict = json.load(f)
# else:
#     label_desc_dict = None

# use the label names/descriptions to replace the label indices
from label_desc import get_label2desc
label2desc = get_label2desc(args.dataset_name)
print(label2desc)


ke = KeywordsExtractor(args.lang)
os.system("mkdir sta_saved_keywords")
kws_dict = ke.global_role_kws_extraction_one_line(
                contents,label_names, 
                label_desc_dict=label2desc,
                output_dir='sta_saved_keywords',
                name=args.dataset_name,
                overwrite=False)
print(kws_dict.keys())

"""
example script:
python sta_extract_kws.py --dataset_name bbc_50
"""