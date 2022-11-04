"""
Selection Text Augmentation (STA)

example script:
python sta_clf.py --dataset_name bbc_50 --method replace --simdict w2v --p 0.1 --n_aug 2
python sta_clf.py --dataset_name bbc_50 --method replace --simdict wordnet --p 0.1 --n_aug 2  # much faster

TODO:
p limit to selected_words only
"""

from STA.text_augmenter import TextAugmenter
import pandas as pd
from tqdm import tqdm
import pickle
import os
import random
random.seed(5)
import argparse
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--dataset_name', type=str, default='bbc_500', help='dataset dir name')
parser.add_argument('--method', type=str, help='replace/delete/insert/positive/mix')
parser.add_argument('--simdict', type=str, default='wordnet', help='dictionary for synonyms searching, wordnet or w2v')
parser.add_argument('--p', type=float, default=0.1, help='prob of the augmentation (augmentation strength)')
parser.add_argument('--n_aug', type=int, default=1, help='how many times to augment')
parser.add_argument('--output_name', type=str, default=None, help='output filename')
args = parser.parse_args()

# read dataset
dataset = pd.read_csv(f'../data_clf/{args.dataset_name}/train.csv')
contents = list(dataset['content'])
labels = list(dataset['label'])

# load sta_saved_keywords
global_kws_dict_path = f'sta_saved_keywords/global_kws_dict_{args.dataset_name}.pkl'
if not os.path.exists(global_kws_dict_path):
    print("Role Keywords not found! Begin extracting the keywords......")
    os.system(f"python sta_extract_kws.py --dataset_name {args.dataset_name}")
    assert os.path.exists(global_kws_dict_path)
    print("Role Keywords extracted!")
with open(global_kws_dict_path, 'rb') as f:
    global_kws_dict = pickle.load(f)
    print("Role Keywords loaded!")

if args.simdict == 'wordnet':
    using_wordnet = True
else:
    using_wordnet = False
TA = TextAugmenter(lang='en', using_wordnet=using_wordnet)


new_contents = []
new_labels = []
puncs = ',.，。!?！？;；、'
punc_list = [w for w in puncs]
for i in range(args.n_aug):
    for content,label in zip(tqdm(contents),labels):
        kws = global_kws_dict[label]
        if args.method == 'mix': # when using 'mix', every sub-operation of eda will be applied once
            new_contents.append(TA.aug_by_replacement(content, args.p, 'selective', 
                                                      selected_words=kws['scw']+kws['fcw']+kws['iw'])) # except ccw
            new_contents.append(TA.aug_by_deletion(content, args.p, 'selective', 
                                                   selected_words=kws['scw']+kws['fcw']+kws['iw']))  # except ccw
            new_contents.append(TA.aug_by_insertion(content, args.p, 'selective', 
                                                    selected_words=kws['ccw']+kws['scw']+kws['iw'])) # except fcw
            new_contents.append(TA.aug_by_selection(content, selected_words=kws['ccw']+kws['iw']+punc_list))
            new_labels += [label] * 4
        if args.method == 'replace':
            new_contents.append(TA.aug_by_replacement(content, args.p, 'selective', 
                                                      selected_words=kws['scw']+kws['fcw']+kws['iw']))
            new_labels.append(label)
        if args.method == 'delete':
            new_contents.append(TA.aug_by_deletion(content, args.p, 'selective', 
                                                   selected_words=kws['scw']+kws['fcw']+kws['iw']))  # except ccw
            new_labels.append(label)
        if args.method == 'insert':
            new_contents.append(TA.aug_by_insertion(content, args.p, 'selective', 
                                                    selected_words=kws['ccw']+kws['scw']+kws['iw']))
            new_labels.append(label)
        if args.method == 'positive':
            new_contents.append(TA.aug_by_selection(content, 
            selected_words=kws['ccw']+random.sample(kws['iw'],k=int(0.8*len(kws['iw'])))+punc_list))
            new_labels.append(label)

augmented_contents = contents + new_contents
augmented_labels = labels + new_labels
assert len(augmented_contents) == len(augmented_labels), 'wrong num'
if args.method == 'mix':
    args.n_aug = args.n_aug * 4
if args.output_name is None:
    args.output_name = f'sta_{args.method}_{args.simdict}_{args.p}_{args.n_aug}'
augmented_dataset = pd.DataFrame({'content':augmented_contents, 'label':augmented_labels})
augmented_dataset.to_csv(f'../data_clf/{args.dataset_name}/{args.output_name}.csv')


print(f'>>> saved to ../data_clf/{args.dataset_name}/{args.output_name}.csv')
print(f'>>> before augmentation: {len(contents)} samples.')
print(f'>>> after augmentation: {len(augmented_contents)} samples.')
