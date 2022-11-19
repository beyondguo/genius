"""
Conditional-Causal Language Modeling (C-CLM) Fine-tuning


- cut the original text into smaller chunks
- prepend the corresponding label prefix the each chunk
- finetune a CLM model (like GPT-2) on the preprocessed data

example script:
python conditional_clm_finetune.py --clm_model_path gpt2 --dataset_name sst2new_50 --num_train_epochs 15

TODO: make sure to use the label names/descriptions
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset as HFDataset
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from accelerate import Accelerator
import math
import pandas as pd
from tqdm import tqdm
import random
random.seed(5)
import os
from collections import defaultdict
import argparse
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--dataset_name', type=str, default='bbc_50', help='dataset dir name')
parser.add_argument('--clm_model_path', type=str, default='distilgpt2', help='gpt2, distilgpt2')
parser.add_argument('--num_train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=8, help='batch size for dataloaders')
parser.add_argument('--max_length', type=int, default=200, help='text length for training CLM')
# parser.add_argument('--device', type=int, default=0, help='cuda device index, if not found, will switch to cpu')
args = parser.parse_args()


# read dataset
# dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data_clf', args.dataset_name, 'train.csv'))
dataset = pd.read_csv(f'../data_clf/{args.dataset_name}/train.csv')
contents = list(dataset['content'])
labels = list(dataset['label'])
# use the label names/descriptions to replace the label indices
from label_desc import get_label2desc
if get_label2desc(args.dataset_name):
    label2desc = get_label2desc(args.dataset_name)
    print(label2desc)
    labels = [label2desc[label] for label in labels]
print(set(labels))

# load base model and tokenizer
base_model_name = args.clm_model_path.split('/')[-1]
model = AutoModelForCausalLM.from_pretrained(args.clm_model_path)
tokenizer = AutoTokenizer.from_pretrained(args.clm_model_path)
tokenizer.pad_token = tokenizer.eos_token
EOS_id = tokenizer.eos_token_id
label2prefix = {label:tokenizer.encode(label+': ', add_special_tokens=False) for label in list(set(labels))}
contents = [c+tokenizer.eos_token for c in contents] # ??
# my_dataset = HFDataset.from_dict({'content':contents, 'label':labels})
# why we don't add label prefix here? cause we want to add label prefix to every chunk
label2contents = defaultdict(list)
for label, content in zip(labels, contents):
    label2contents[label].append(content)


lm_data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


max_length = args.max_length
all_chunk_res = []
for label in label2contents:
    chunk_size = max_length - len(label2prefix[label]) # e.g. prefix=[1371,131], chun_size=200-2=198
    tokenize_res = tokenizer(label2contents[label]) # don not use `padding` or set `max_length` here!
    # concat all the list in each column
    concat_res = {k:sum(tokenize_res[k], []) for k in tokenize_res} 
    concat_lm_res = lm_data_collator([concat_res]) # columns: input_ids, attention_mask, labels
    # split the concatenated list into chunks
    total_length = len(concat_lm_res['input_ids'][0])
    chunk_res = defaultdict(list)
    if total_length < chunk_size: # the current label doesn't have enough data
        continue
    for k,t in concat_lm_res.items():
        t = t[0].tolist()
        for i in range(total_length // chunk_size): # just drop the last chunk
            # select the chunk
            chunk = t[i * chunk_size : (i+1) * chunk_size]
            if k == 'input_ids':
                # prepend the label prefix
                chunk = label2prefix[label] + chunk
            elif k == 'labels':
                # we don't predict the label prefix
                chunk = [-100] * len(label2prefix[label]) + chunk
            else: #  attention_mask
                chunk = [1] * len(label2prefix[label]) + chunk
            chunk_res[k].append(chunk)
    all_chunk_res.append(chunk_res)

train_dict = {}
for k in all_chunk_res[0].keys():
    train_dict[k] = sum([each[k] for each in all_chunk_res], [])

training_dataset = HFDataset.from_dict(train_dict)

num_train_epochs = args.num_train_epochs
batch_size = args.batch_size

def collate_fn(examples):
    res = defaultdict(list)
    for k in examples[0]:
        for example in examples:
            res[k].append(example[k])
    return res
train_dataloader = DataLoader(training_dataset,batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


accelerator = Accelerator()
optimizer = AdamW(model.parameters(), lr=5e-5)
model, optimizer = accelerator.prepare(model, optimizer)

num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * len(train_dataloader)

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)


print('Start training...')
for e in range(num_train_epochs):
    # train:
    _ = model.train()
    print(f'>>> Epoch {e+1}')
    losses = []
    for batch in train_dataloader:
        batch = {k:torch.tensor(v).to(accelerator.device) for k,v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        losses.append(loss)
    try:
        mean_loss = sum(losses)/len(losses)  # this loss is not precise, just for monitoring
        print(f'   >>> approx average loss: {mean_loss}')
    except Exception as e:
        print(e)
    


# Save model
args.output_dir = f'../saved_models/c-clm/{args.dataset_name}_{base_model_name}'
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
tokenizer.save_pretrained(args.output_dir)



# use a pipeline to generate new text:
# from transformers import pipeline
# text_gen = pipeline("text-generation", model=args.output_dir)
# for label in list(set(labels)):
#     print(f'{label}')
#     print(f">>> {text_gen(label+': ', pad_token_id=tokenizer.eos_token_id)[0]['generated_text']}")