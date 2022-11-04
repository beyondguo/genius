"""
Easy Data Augmentation (EDA)

example script:
python eda_clf.py --dataset_name bbc_50 --method mix --simdict wordnet --p 0.1 --n_aug 4
python eda_clf.py --dataset_name bbc_50 --method mix --simdict w2v --p 0.1 --n_aug 2
"""

from STA.text_augmenter import TextAugmenter
import pandas as pd
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--dataset_name', type=str, default='bbc_500', help='dataset dir name, in data/')
parser.add_argument('--method', type=str, help='replace/delete/insert/swap/mix')
parser.add_argument('--simdict', type=str, default='wordnet', help='dictionary for synonyms searching, wordnet or w2v')
parser.add_argument('--p', type=float, default=0.1, help='prob of the augmentation (augmentation strength)')
parser.add_argument('--n_aug', type=int, default=1, help='how many times to augment')
parser.add_argument('--output_name', type=str, default=None, help='output filename')
args = parser.parse_args()

# read dataset
dataset = pd.read_csv(f'../data_clf/{args.dataset_name}/train.csv')
contents = list(dataset['content'])
labels = list(dataset['label'])

if args.simdict == 'wordnet':
    using_wordnet = True
else:
    using_wordnet = False
TA = TextAugmenter(lang='en', using_wordnet=using_wordnet)


new_contents = []
new_labels = []
for i in range(args.n_aug):
    for content,label in zip(tqdm(contents),labels):
        if args.method == 'mix': # when using 'mix', every sub-operation of eda will be applied once
            new_contents.append(TA.aug_by_replacement(content, args.p))
            new_contents.append(TA.aug_by_deletion(content, args.p))
            new_contents.append(TA.aug_by_insertion(content, args.p))
            new_contents.append(TA.aug_by_swap(content, args.p))
            new_labels += [label] * 4
        if args.method == 'replace':
            new_contents.append(TA.aug_by_replacement(content, args.p))
            new_labels.append(label)
        if args.method == 'delete':
            new_contents.append(TA.aug_by_deletion(content, args.p))
            new_labels.append(label)
        if args.method == 'insert':
            new_contents.append(TA.aug_by_insertion(content, args.p))
            new_labels.append(label)
        if args.method == 'swap':
            new_contents.append(TA.aug_by_swap(content, args.p))
            new_labels.append(label)

augmented_contents = contents + new_contents
augmented_labels = labels + new_labels
assert len(augmented_contents) == len(augmented_labels), 'wrong num'
if args.method == 'mix':
    args.n_aug = args.n_aug * 4
if args.output_name is None:
    args.output_name = f'eda_{args.method}_{args.simdict}_{args.p}_{args.n_aug}'
augmented_dataset = pd.DataFrame({'content':augmented_contents, 'label':augmented_labels})
augmented_dataset.to_csv(f'../data_clf/{args.dataset_name}/{args.output_name}.csv')


print(f'>>> saved to ../data_clf/{args.dataset_name}/{args.output_name}.csv')
print(f'>>> before augmentation: {len(contents)} samples.')
print(f'>>> after augmentation: {len(augmented_contents)} samples.')
