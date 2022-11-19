"""
example:
python aug_filter_clf.py --dataset_name bbc_50 --backbone distilbert-base-cased --aug_file_name sega_promptFalse_sega-base-t4-as_aug4 --threshold 0.7
"""
import sys
sys.path.append('../')
from my_dataset import get_dataloader
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding
import argparse
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--dataset_name', type=str, default='bbc_50', help='dataset dir name')
parser.add_argument('--backbone', type=str, default='distilbert-base-cased', help='base model')
# parser.add_argument('--clf_model_path', type=str, help='base model')
parser.add_argument('--aug_file_name', type=str, help='aug_file_name, name before .csv')
parser.add_argument('--threshold', type=float, default=0.7, help='threshold for filtering')
parser.add_argument('--filter_all', action='store_true', default=False, help='if set, will filter all the samples, including the original ones')
args = parser.parse_args()

orig_train_num = int(args.dataset_name.split('_')[-1])
aug_file_path = f'../data_clf/{args.dataset_name}/{args.aug_file_name}.csv'
train_df = pd.read_csv(aug_file_path)
labels = list(train_df['label'])
unique_labels = sorted(list(set(labels)))
label2idx = {unique_labels[i]: i for i in range(len(unique_labels))}
idx2label = {label2idx[label]: label for label in label2idx}


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu',1)
print('>>> ',device)

clf_model = AutoModelForSequenceClassification.from_pretrained(args.backbone, num_labels=len(unique_labels))
clf_model.load_state_dict(torch.load(f'../saved_models/{args.dataset_name}_{args.backbone}_train.pkl')) # the non-aug model
clf_model.to(device)
tokenizer = AutoTokenizer.from_pretrained(args.backbone)

bz = 16
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_dataloader = get_dataloader(
    aug_file_path, tokenizer, label2idx, 200, bz, 
    data_collator, shuffle=False) # must set `shuffle=False` to keep the original order



clf_model.eval()
# all_probs = []
all_true_label_probs = []
i = 0
for batch in tqdm(train_dataloader):
    batch = {k:v.to(device) for k,v in batch.items()}
    logits = clf_model(**batch).logits
    probs = torch.softmax(logits, dim=1)
    # all_probs.append(probs.cpu())
    label_ids = [label2idx[label] for label in labels[i*bz :(i+1)*bz]]
    label_ids = torch.LongTensor([[idx] for idx in label_ids])
    true_label_probs = probs.gather(1, label_ids.to(device))
    # print(true_label_probs.shape)
    # print(true_label_probs.view(-1,).tolist())
    all_true_label_probs += true_label_probs.view(-1,).tolist()
    i += 1

# all_probs_matrix = torch.cat(all_probs)
# label_ids = [label2idx[label] for label in labels]
# label_ids = torch.LongTensor([[idx] for idx in label_ids])
# true_label_probs = all_probs_matrix.gather(1, label_ids.to(device))


keep_ids = []
if args.filter_all:
    print('>>> filtering all samples...')
    for i,prob in enumerate(all_true_label_probs):
        if prob > args.threshold:
            keep_ids.append(i)
    print('keep ratio: ', len(keep_ids)/len(labels))
    print('keep ratio in aug-samples: ', len(keep_ids[orig_train_num:])/len(labels[orig_train_num:]))
else:
    # only filter augmented samples
    # using this option, you must make sure the augmented samples are append to the original ones, without any shuffing
    print('>>> filtering on the augmented samples...')
    keep_ids += list(range(orig_train_num))
    for i,prob in enumerate(all_true_label_probs[orig_train_num:]):
        if prob > args.threshold:
            keep_ids.append(i)
    print('keep ratio: ', len(keep_ids)/len(labels))
    print('keep ratio in aug-samples: ', len(keep_ids[orig_train_num:])/len(labels[orig_train_num:]))

keep_df = train_df.iloc[keep_ids,:].reset_index(drop=True)

for i in range(len(keep_ids)):
    c1 = keep_df.iloc[i,:]['content'][:100]
    l1 = keep_df.iloc[i,:]['label']
    c2 = train_df.iloc[keep_ids[i],:]['content'][:100]
    l2 = train_df.iloc[keep_ids[i],:]['label']
    assert c1 == c2
    assert l1 == l2


filtered_file_path = f'../data_clf/{args.dataset_name}/{args.aug_file_name}_filtered_{args.threshold}.csv'
keep_df.to_csv(filtered_file_path)
print(f'>>> saved to: {filtered_file_path}')