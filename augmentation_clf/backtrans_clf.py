"""
Augmentation by Back-Translation.

example script:
python backtrans_clf.py --dataset_name bbc_50 --inter_langs de-zh --n_aug 2
"""


from transformers import pipeline
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
import random
random.seed(5)
import argparse
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--dataset_name', type=str, default='bbc_500', help='dataset dir name')
parser.add_argument('--inter_langs', type=str, help='support zh,ru,de,es, concat by -, max: de-zh-fr-es')
parser.add_argument('--n_aug', type=int, default=1, help='how many times to augment')
parser.add_argument('--output_name', type=str, default=None, help='output filename')
args = parser.parse_args()


class MyDataset(Dataset):
    def __init__(self, contents):
        self.contents = contents
    def __len__(self):
        return len(self.contents)
    def __getitem__(self, i):
        return self.contents[i]

# read dataset
dataset = pd.read_csv(f'../data_clf/{args.dataset_name}/train.csv')
contents = list(dataset['content'])
labels = list(dataset['label'])


# translator
lang2model = {
    'de': ('Helsinki-NLP/opus-mt-en-de', 'Helsinki-NLP/opus-mt-de-en'),
    'zh': ('Helsinki-NLP/opus-mt-en-zh', 'Helsinki-NLP/opus-mt-zh-en'),
    'fr': ('Helsinki-NLP/opus-mt-en-fr', 'Helsinki-NLP/opus-mt-fr-en'),
    'es': ('Helsinki-NLP/opus-mt-en-es', 'Helsinki-NLP/opus-mt-es-en'),
}
inter_langs = args.inter_langs.split('-')
print(inter_langs)
translators = {}
for i,lang in enumerate(inter_langs):
    translators[lang] ={
        'forward':pipeline("translation", model=lang2model[lang][0], device=i),
        'backward':pipeline("translation", model=lang2model[lang][1], device=i)
        }




# forward:
forward_dataset = MyDataset(contents)
inter_contents_list = []
used_langs = []
for i in range(args.n_aug):
    inter_contents = []
    random_lang = random.choice(inter_langs)
    translator = translators[random_lang]
    print(f"current inter language: {random_lang}")
    used_langs.append(random_lang)
    for out in tqdm(translator['forward'](forward_dataset, num_beams=5, do_sample=True, truncation=True, max_length=200, batch_size=32)):
        inter_text = out[0]['translation_text']
        inter_contents.append(inter_text)
    inter_contents_list.append(inter_contents)

# backward:
backtrans_contents = []
for inter_contents,lang in zip(inter_contents_list, used_langs):
    backward_dataset = MyDataset(inter_contents)
    translator = translators[lang]
    for out in tqdm(translator['backward'](backward_dataset, num_beams=5, do_sample=True, truncation=True, max_length=200, batch_size=32)):
        back_text = out[0]['translation_text']
        backtrans_contents.append(back_text)


new_labels = labels * args.n_aug
augmented_contents = contents + backtrans_contents
augmented_labels = labels + new_labels
assert len(augmented_contents) == len(augmented_labels), 'wrong num'
if args.output_name is None:
    args.output_name = f'backtrans_{args.inter_langs}_{args.n_aug}'
augmented_dataset = pd.DataFrame({'content':augmented_contents, 'label':augmented_labels})
augmented_dataset.to_csv(f'../data_clf/{args.dataset_name}/{args.output_name}.csv')


print(f'>>> saved to ../data_clf/{args.dataset_name}/{args.output_name}.csv')
print(f'>>> before augmentation: {len(contents)} samples.')
print(f'>>> after augmentation: {len(augmented_contents)} samples.')
