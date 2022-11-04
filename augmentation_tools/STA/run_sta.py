
import argparse
import pandas as pd
import jieba
from tqdm import tqdm
import os
import pickle
from text_augmenter import TextAugmenter
from keywords_extractor import role_kws_extraction_single

parser = argparse.ArgumentParser(allow_abbrev=False)
# 常用参数：
parser.add_argument('--dataset', type=str, default='bbc_500', help='dataset dir name, in data/')
parser.add_argument('--lang', type=str, default='en', help='language, en or zh')
parser.add_argument('--strategy', type=str, default='local', help='local or global')
parser.add_argument('--methods', type=str, default='se', help='methods of augmentation, join by ","')
parser.add_argument('--p', type=float, default=0.1, help='prob of the augmentation')
parser.add_argument('--bar', type=str, default='Q2', help='the bar of extracting role keywords, Q1, Q2, Q3 for three quartiles')
parser.add_argument('--n_aug', type=int, default=1, help='how many times to augment')
parser.add_argument('--best_practice', default=False, action='store_true')

parser.add_argument('--ablation_without',type=str, default=None, help='lr or ls')



args = parser.parse_args()
methods = args.methods.split(',')
for method in methods:
    assert method in ['re','in','sw','de','se'], '"%s" is not supported yet'%method

dataset_name = args.dataset  # /data/...
output_dir = f"data/{dataset_name}/selective_{args.strategy}_{'_'.join(methods)}_{args.p}_{args.bar}_{args.n_aug}/"
if args.ablation_without is not None:
    output_dir = f"data/ablation_study/{dataset_name}/selective_{args.strategy}_{'_'.join(methods)}_{args.p}_{args.bar}_{args.n_aug}/"
if os.path.exists(output_dir) == False:
    os.makedirs(output_dir)

train_path = f'data/{dataset_name}/train.csv'
global_lr_dict_path = f'saved_keywords/global_lr_dict_{dataset_name}.pkl'
global_ls_dict_path = f'saved_keywords/global_ls_dict_{dataset_name}.pkl'
global_kws_dict_path = f'saved_keywords/global_kws_dict_{dataset_name}.pkl'

assert os.path.exists(global_lr_dict_path) and os.path.exists(global_ls_dict_path), "file not exists!"
assert os.path.exists(global_kws_dict_path), 'file not exists!'

raw_train_df = pd.read_csv(train_path)
 # 处理空值问题
raw_train_df = raw_train_df.dropna()
raw_train_df = raw_train_df[raw_train_df.content != '']

texts = list(raw_train_df['content'])
labels = list(raw_train_df['label'])
with open(global_lr_dict_path, 'rb') as f:
    global_lr_dict = pickle.load(f)
with open(global_ls_dict_path, 'rb') as f:
    global_ls_dict = pickle.load(f)
with open(global_kws_dict_path, 'rb') as f:
    global_kws_dict = pickle.load(f)

TA = TextAugmenter(args.lang)

puncs = ',.，。!?！？;；、'
punc_list = [w for w in puncs]
special_tokens = ",./;\`~<>?:\"，。/；‘’“”、｜《》？～· \n[]{}【】「」（）()0123456789０１２３４５６７８９" \
            "，．''／；\｀～＜＞？：＂,。／;‘’“”、|《》?~·　\ｎ［］｛｝【】「」("")（） "
stop_words = TA.stop_words
skip_words = [t for t in special_tokens] + stop_words

# 合并数据集，先把原始样本加入
mix_contents = []
mix_labels = []
mix_contents += texts
mix_labels += labels

print_info = False
for method in methods:
    aug_filename = output_dir+f'train_{method}.csv'
    augmented_texts = []
    for i in range(args.n_aug): # augment multiple times
        for text,label in zip(tqdm(texts), labels):
            words = TA.tokenizer(text)
            # 使用局部关键词，即从当前文本中
            if args.strategy == 'local':
                kws = role_kws_extraction_single(words, label, global_ls_dict, global_lr_dict, bar=args.bar, skip_words=skip_words)
            # 使用全局关键词
            elif args.strategy == 'global':
                kws = global_kws_dict[label]
                
            if args.ablation_without == 'lr': 
                print(f'>>> ABLATION STUDY: WITHOUT [lr]')
                # 消融实验，不再使用label correlation
                # 那么fcw, iw就合并为iw；ccw,scw就合并为ccw
                # fcw， scw皆为空
                kws['ccw'] = kws['ccw'] + kws['scw']
                kws['iw'] = kws['iw'] + kws['fcw']
                kws['fcw'] = []
                kws['scw'] = []
            if args.ablation_without == 'ls':
                print(f'>>> ABLATION STUDY: WITHOUT [ls]')
                # 消融实验，不再使用label correlation
                # 那么ccw,fcw就合并为ccw；iw, scw就合并为iw
                # fcw， scw皆为空
                kws['ccw'] = kws['ccw'] + kws['fcw']
                kws['iw'] = kws['iw'] + kws['scw']
                kws['fcw'] = []
                kws['scw'] = []
                

            if method == 'de':
                new_words = TA.aug_by_deletion(text, args.p, 'selective', print_info=print_info,
                                               selected_words=kws['scw']+kws['fcw']+kws['iw'])  # except ccw
            elif method == 're':
                new_words = TA.aug_by_replacement(text, args.p, 'selective', print_info=print_info,
                                                  selected_words=kws['scw']+kws['fcw']+kws['iw'])  # except ccw
            elif method == 'in':
                new_words = TA.aug_by_insertion(text, args.p, 'selective', print_info=print_info,
                                                given_words=kws['ccw']+kws['scw']+kws['iw'])  # except fcw
            elif method == 'sw':
                new_words = TA.aug_by_swap(text, args.p, 'selective', print_info=print_info,
                                           selected_words=kws['iw'])  # iw better
            elif method == 'se':
                new_words = TA.aug_by_selection(text, print_info=print_info,
                                                selected_words=kws['ccw']+kws['iw']+punc_list)
            else:
                raise NotImplementedError()

            joint_str = ' ' if args.lang == 'en' else ''
            new_text = joint_str.join(new_words)
            for punc in puncs: # 处理上面的拼接造成的一些小问题
                new_text = new_text.replace(' '+punc, punc)
            augmented_texts.append(new_text)
    # 每种方法先单独保存一份：
    new_df = pd.DataFrame({'content': texts+augmented_texts, 'label': labels*(args.n_aug+1)})
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

