"""
Conditional-Masked Language Modeling (C-MLM) Fine-tuning

- cut the original text into smaller chunks
- apply random masking on these chunks
- prepend the corresponding label prefix the each chunk
- finetune a MLM model on the processes data

Fintune a MLM model, by prepending the label before the input text.

example script:
python conditional_mlm_finetune.py --dataset_name sst2new_50 --mlm_model_path roberta-base --num_train_epochs 15

TODO: make sure to use the label names/descriptions
"""

from transformers import AutoModelForMaskedLM, AutoTokenizer
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
from collections import defaultdict
import argparse
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--dataset_name', type=str, default='bbc_50', help='dataset dir name')
parser.add_argument('--mlm_model_path', type=str, default='distilbert-base-cased', help='mlm model name (from the huggingface hub) or local model path')
parser.add_argument('--p', type=float, default=0.15, help='augmentation strength')
parser.add_argument('--num_train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=8, help='batch size for dataloaders')
parser.add_argument('--device', type=int, default=0, help='cuda device index, if not found, will switch to cpu')
args = parser.parse_args()


# read dataset
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

# load mlm model
base_model_name = args.mlm_model_path.split('/')[-1]
model = AutoModelForMaskedLM.from_pretrained(args.mlm_model_path)
tokenizer = AutoTokenizer.from_pretrained(args.mlm_model_path)
SEP_id = tokenizer.sep_token_id
MASK_id = tokenizer.mask_token_id


label2contents = defaultdict(list)
for label, content in zip(labels, contents):
    label2contents[label].append(content)


label2prefix = {label:tokenizer.encode(label+': ', add_special_tokens=False) for label in list(set(labels))}


mlm_data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=args.p)
max_length = 200

def get_random_mask_dataset():
    # this method will produce "fixed" masked training set
    # however, during training, we would like different masking with every epoch
    # therefore, this method should be called in every epoch
    all_chunk_res = []
    for label in label2contents:
        chunk_size = max_length - len(label2prefix[label]) # e.g. prefix=[1371,131], chun_size=200-2=198
        tokenize_res = tokenizer(label2contents[label]) # don not use `padding` or set `max_length` here!
        # concat all the list in each column
        concat_res = {k:sum(tokenize_res[k], []) for k in tokenize_res} 
        concat_mlm_res = mlm_data_collator([concat_res]) # columns: input_ids, attention_mask, labels

        # split the concatenated list into chunks
        total_length = len(concat_mlm_res['input_ids'][0])
        chunk_res = defaultdict(list)
        for k,t in concat_mlm_res.items():
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
    return training_dataset


def collate_fn(examples):
    res = defaultdict(list)
    for k in examples[0]:
        for example in examples:
            res[k].append(example[k])
    return res



num_train_epochs = args.num_train_epochs
batch_size = args.batch_size
eval_dataset = get_random_mask_dataset()
eval_dataloader = DataLoader(eval_dataset,batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

accelerator = Accelerator()
optimizer = AdamW(model.parameters(), lr=5e-5)
model, optimizer = accelerator.prepare(model, optimizer)

num_update_steps_per_epoch = len(eval_dataloader)
num_training_steps = num_train_epochs * len(eval_dataloader)

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)


# check the perplexity of original pretrained model
_ = model.eval()
losses = []
for batch in eval_dataloader:
    batch = {k:torch.tensor(v).to(accelerator.device) for k,v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    loss = outputs.loss
    losses.append(loss)
loss_mean = sum(losses)/len(losses)  # this loss is not precise, just for monitoring
try:
    perplexity = math.exp(loss_mean)
except:
    perplexity = float("inf")
print(f' ---Original Test perplexity: {perplexity}')

print('Start training...')
for e in range(num_train_epochs):
    # train:
    _ = model.train()
    print(f'>>> Epoch {e+1}')
    training_dataset = get_random_mask_dataset()
    dataloader = DataLoader(training_dataset,batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    # dataloader = accelerator.prepare(dataloader)
    for batch in dataloader:
        batch = {k:torch.tensor(v).to(accelerator.device) for k,v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        print(f'   >>> batch loss: {loss}')
    
    # evaluation:
    _ = model.eval()
    losses = []
    for batch in eval_dataloader:
        batch = {k:torch.tensor(v).to(accelerator.device) for k,v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        losses.append(loss)
    loss_mean = sum(losses)/len(losses)  # this loss is not precise, just for monitoring
    try:
        perplexity = math.exp(loss_mean)
    except:
        perplexity = float("inf")
    print(f' ---Test perplexity: {perplexity}')
    
    
# Save model
args.output_dir = f'../saved_models/c-mlm/{args.dataset_name}_{base_model_name}'
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
tokenizer.save_pretrained(args.output_dir)



# new_model = AutoModelForMaskedLM.from_pretrained(args.output_dir)
# from transformers import pipeline
# mask_filler = pipeline("fill-mask", model=args.output_dir)
