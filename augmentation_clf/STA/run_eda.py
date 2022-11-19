from text_augmenter import TextAugmenter
import argparse
import pandas as pd
import jieba
from tqdm import tqdm
import os
parser = argparse.ArgumentParser(allow_abbrev=False)
# 常用参数：
parser.add_argument('--dataset', type=str, default='bbc_500', help='dataset dir name, in data/')
parser.add_argument('--lang', type=str, default='en', help='language, en or zh')
parser.add_argument('--methods', type=str, help='methods of augmentation, join by ","')
parser.add_argument('--p', type=float, default=0.1, help='prob of the augmentation')
parser.add_argument('--n_aug', type=int, default=1, help='how many times to augment')



args = parser.parse_args()
TA = TextAugmenter(args.lang, using_wordnet=True)
methods = args.methods.split(',')
for method in methods:
    assert method in ['re','in','sw','de'], '"%s" is not supported yet'%method

dataset_name = args.dataset  # /data/...
# output_dir = f"data/{dataset_name}/random_{'_'.join(methods)}_{args.p}_{args.n_aug}/"
# use wordnet
output_dir = f"data/{dataset_name}/random_{'_'.join(methods)}_{args.p}_{args.n_aug}_wordnet/"
if os.path.exists(output_dir) == False:
    os.makedirs(output_dir)
    
train_path = 'data/'+dataset_name+'/'+'train.csv'
raw_train_df = pd.read_csv(train_path)
 # 处理空值问题
raw_train_df = raw_train_df.dropna()
raw_train_df = raw_train_df[raw_train_df.content != '']

texts = list(raw_train_df['content'])
labels = list(raw_train_df['label'])

puncs = ',.，。!?！？;；、'
mix_contents = []
mix_labels = []
mix_contents += texts
mix_labels += labels

for method in methods:
    aug_filename = output_dir+f'train_{method}.csv'
    augmented_texts = []
    for text in tqdm(texts):
        if method == 'de':
            new_words = TA.aug_by_deletion(text, args.p, 'random')
        elif method == 're':
            new_words = TA.aug_by_replacement(text, args.p, 'random')
        elif method == 'in':
            new_words = TA.aug_by_insertion(text, args.p, 'random')
        elif method == 'sw':
            new_words = TA.aug_by_swap(text, args.p, 'random')
        else:
            raise NotImplementedError()
        joint_str = ' ' if args.lang == 'en' else ''
        new_text = joint_str.join(new_words)
        for punc in puncs: # 处理上面的拼接造成的一些小问题
            new_text = new_text.replace(' '+punc, punc)
        augmented_texts.append(new_text)
    
    # 每种方法先单独保存一份：
    new_df = pd.DataFrame({'content': augmented_texts, 'label': labels*args.n_aug})
    new_df.to_csv(aug_filename)
    print('saved to %s'%aug_filename)
    # 然后加入到合并数据集中：
    mix_contents += augmented_texts
    mix_labels += labels*args.n_aug
    

mix_filename = output_dir+'train_mix.csv'
mix_df = pd.DataFrame({'content': mix_contents, 'label': mix_labels})
mix_df.to_csv(mix_filename)
print('saved to %s'%mix_filename)

print(f'>>> before augmentation: {len(texts)}')
print(f'>>> after augmentation: {len(mix_contents)}')