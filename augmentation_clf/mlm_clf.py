"""
Augmentation by Masked Language Modeling (MLM)

example script:
python mlm_clf.py --dataset_name bbc_50 --mlm_model_path distilbert-base-cased --p 0.1 --topk 5 --n_aug 2
TODO: support whole-word masking (wwm)
"""

from transformers import pipeline, AutoModelForMaskedLM, AutoTokenizer
from transformers import default_data_collator, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import random
random.seed(5)
import argparse
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--dataset_name', type=str, default='bbc_50', help='dataset dir name')
parser.add_argument('--mlm_model_path', type=str, default='distilbert-base-cased', help='mlm model name (from the huggingface hub) or local model path')
parser.add_argument('--p', type=float, default=0.15, help='augmentation strength')
parser.add_argument('--n_aug', type=int, default=1, help='how many times to augment')
parser.add_argument('--topk', type=int, default=5, help='topk sampling of MLM predictoin')
parser.add_argument('--output_name', type=str, default=None, help='output filename')
parser.add_argument('--device', type=int, default=0, help='cuda device index, if not found, will switch to cpu')
args = parser.parse_args()


# read dataset
dataset = pd.read_csv(f'../data_clf/{args.dataset_name}/train.csv')
contents = list(dataset['content'])
labels = list(dataset['label'])

try:
    device = torch.device('cuda', index=args.device)
except Exception as e:
    print(e)
    device = torch.device('cpu')
print(device)

# load mlm model
model = AutoModelForMaskedLM.from_pretrained(args.mlm_model_path)
tokenizer = AutoTokenizer.from_pretrained(args.mlm_model_path)
MASK_id = tokenizer.mask_token_id

# mask inputs
def tokenize_function(examples):
    res = tokenizer(examples['content'])
    res['label'] = examples['label']
    # add a new column "word_ids" for later use
    # "input_ids" means the ids for each token (including the sub-tokens)
    # "word_ids" means the ids for each word, if a word in tokenized into several sub-words, they will share a same "word id"
    res["word_ids"] = [res.word_ids(i) for i in range(len(res["input_ids"]))]
    return res

# tokenize_res = tokenizer(contents, max_length=200, padding=True, truncation=True, return_tensors='pt')
# tokenize_res["word_ids"] = [tokenize_res.word_ids(i) for i in range(len(tokenize_res["input_ids"]))]


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=args.p)
# example_batch = [{k:tokenize_res[k][i] for k in tokenize_res if k != 'word_ids'} for i in range(3)]
# mask_res = data_collator(example_batch)


class TokenizedDataset:
    def __init__(self, contents):
        self.contents = contents
        self.encodings = tokenizer(contents, max_length=200, padding=True, truncation=True, return_tensors='pt')
    def __getitem__(self, idx):
        item = {k:v[idx] for k,v in self.encodings.items()}
        return item
    def __len__(self):
        return len(self.encodings['input_ids'])

tokenized_dataset = TokenizedDataset(contents)
masked_dataloader = DataLoader(tokenized_dataset, batch_size=4, shuffle=False, collate_fn=data_collator)


model.to(device)
new_contents = []
for _ in range(args.n_aug):
    for batch in tqdm(masked_dataloader):
        batch_size = len(batch['input_ids'])
        batch = {k:v.to(device) for k,v in batch.items() if k != 'labels'}
        logits = model(**batch).logits
        # find the location of [MASK] tokens for each sample in the batch
        for i in range(batch_size):
            mask_token_idx = torch.where(batch['input_ids'][i] == MASK_id)[0] # torch.Size([num_mask])
            mask_token_logits = logits[i][mask_token_idx, :] # torch.Size([num_mask, vocab_size])
            topk_candidate_tokens = torch.topk(mask_token_logits, k=args.topk, dim=1).indices.tolist() # indices.shape: torch.Size([num_mask, topk])
            # replace the [MASK] tokens one by one:
            new_input_ids = batch['input_ids'][i].tolist()[:] # the i-th sample
            for idx,candidates in zip(mask_token_idx, topk_candidate_tokens):
                new_input_ids[idx] = random.choice(candidates)
            # new_input_ids_list.append(new_input_ids)
            new_contents.append(tokenizer.decode(new_input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True))

assert len(new_contents) == len(contents) * args.n_aug, "wrong num"

new_labels = labels * args.n_aug
augmented_contents = contents + new_contents
augmented_labels = labels + new_labels
assert len(augmented_contents) == len(augmented_labels), 'wrong num'
if args.output_name is None:
    args.output_name = f"mlm_{args.mlm_model_path.split('/')[-1]}_p{args.p}_top{args.topk}_aug{args.n_aug}"
augmented_dataset = pd.DataFrame({'content':augmented_contents, 'label':augmented_labels})
augmented_dataset.to_csv(f'../data_clf/{args.dataset_name}/{args.output_name}.csv')


print(f'>>> saved to ../data_clf/{args.dataset_name}/{args.output_name}.csv')
print(f'>>> before augmentation: {len(contents)} samples.')
print(f'>>> after augmentation: {len(augmented_contents)} samples.')
