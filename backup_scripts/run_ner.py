# python -m torch.distributed.launch --nproc_per_node 1 --use_env run_ner.py
# CUDA_VISIBLE_DEVICES=0 python run_ner.py

from numpy import True_
import pandas as pd
from tqdm.auto import tqdm
import logging
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
import datasets
import transformers
from datasets import load_dataset, load_metric, concatenate_datasets, Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification, get_scheduler, set_seed
import random

set_seed(1)

logger = logging.getLogger(__name__)

accelerator = Accelerator()
logger.info(accelerator.state)
logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
if accelerator.is_local_main_process:
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_warning()
else:
    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()


# dataset
dataset_name = 'wikiann'
raw_datasets = load_dataset(dataset_name,'en')

label_names = raw_datasets["train"].features["ner_tags"].feature.names
id2label = {str(i): label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}

model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            ### 方案1：后面的subword跟前面是同一个实体
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            ### 方案2：后面的subword的label设为-100，不参与计算
            # label = -100
            new_labels.append(label)
    return new_labels


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

def tokenize_and_align_labels_block_others(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        labels = [-100 if label == 0 and random.uniform(0,1)<BLOCK_P else label for label in labels]
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


def show(dataset, i, print_to_file=None):
    words = dataset['tokens'][i]
    labels = dataset['ner_tags'][i]
    line1 = ""
    line2 = ""
    for word, label in zip(words, labels):
        full_label = label_names[label]
        max_length = max(len(word), len(full_label))
        line1 += word + " " * (max_length - len(word) + 1)
        line2 += full_label + " " * (max_length - len(full_label) + 1)
    print(line1)
    print(line2)
    if print_to_file is not None:
        print('[%s]--------------------------------------'%i)
        print(line1,file=print_to_file)
        print(line2,file=print_to_file)

tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)

# adding the augmented dataset========================================================================================
output_dir = 'saved_models/ner/base-500'
DO_SAVE = False
num_train = 100
TRAIN_AUG = True
BLOCK_O = True
BLOCK_P = 1
print('********** Main Params: **********')
print(f'>>>>>>>>>>>> num_train: {num_train}')
print(f'>>>>>>>>>>>> with aug: {TRAIN_AUG}')
print(f'>>>>>>>>>>>> block: {BLOCK_O}')
print(f'>>>>>>>>>>>> block_p: {BLOCK_P}')

if num_train:
    train_dataset = tokenized_datasets["train"].select(range(num_train))
else:
    train_dataset = tokenized_datasets["train"]

if TRAIN_AUG:
    # aug_file_name = 'ner_data/conll03-500-MELM.pkl' # other
    aug_file_name = 'ner_data/wikiann-100-S2T-aug1_long.pkl' # k2t-frame
    # -----------------
    # aug_file_name2 = 'ner_data/conll03-500-K2T-frame-aug1_long_v5_filtered_1.pkl' # k2t-frame
    # df1 = pd.read_pickle(aug_file_name1)
    # df2 = pd.read_pickle(aug_file_name2)
    # augmented_dataset1 = Dataset.from_pandas(df1)
    # augmented_dataset2 = Dataset.from_pandas(df2)
    # # augmented_dataset = concatenate_datasets([augmented_dataset1, augmented_dataset2])
    # # augmented_dataset = augmented_dataset2
    # tokenized_aug_datasets1 = augmented_dataset1.map(
    #         tokenize_and_align_labels,batched=True,
    #         remove_columns=augmented_dataset1.column_names,)
    # tokenized_aug_datasets2 = augmented_dataset2.map(
    #         tokenize_and_align_labels_block_others,batched=True,
    #         remove_columns=augmented_dataset2.column_names,)
    # tokenized_aug_datasets = concatenate_datasets([tokenized_aug_datasets1,tokenized_aug_datasets2])

    # print(tokenized_aug_datasets)
    # -----------------
    df = pd.read_pickle(aug_file_name)
    print(f'>>>>>>>>>>>> Aug file: {aug_file_name}')
    print(f'>>>>>>>>>>>> Aug size: {len(df)}')
    augmented_dataset = Dataset.from_pandas(df)

    if BLOCK_O:
        tokenized_aug_datasets = augmented_dataset.map(
            tokenize_and_align_labels_block_others,batched=True,
            remove_columns=augmented_dataset.column_names,)
    else:
        tokenized_aug_datasets = augmented_dataset.map(
            tokenize_and_align_labels,batched=True,
            remove_columns=augmented_dataset.column_names,)
    if num_train:
        train_dataset = concatenate_datasets(
        [tokenized_datasets["train"].select(range(num_train)),tokenized_aug_datasets])
    else:
        train_dataset = concatenate_datasets(
        [tokenized_datasets["train"],tokenized_aug_datasets])

# prepare for training
batch_size = 16
num_train_epochs = 40

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# use different training set
train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    collate_fn=data_collator,
    batch_size=batch_size,)


eval_dataloader = DataLoader(
    dataset = tokenized_datasets["validation"].select(range(num_train)) if num_train else tokenized_datasets["validation"], 
    collate_fn=data_collator, batch_size=batch_size)

test_dataloader = DataLoader(
    tokenized_datasets["test"], collate_fn=data_collator, batch_size=batch_size)


model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,)



optimizer = AdamW(model.parameters(), lr=2e-5)

model, optimizer, train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader, test_dataloader)


num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,)

def postprocess(predictions, labels):
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)]
    return true_labels, true_predictions


metric = load_metric("seqeval")
def dataloader_evaluation(model, dataloader, return_metric='overall_f1',verbose=False):
    model.eval()
    for batch in dataloader:
        with torch.no_grad():
            outputs = model(**batch)

        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]

        # Necessary to pad predictions and labels for being gathered
        predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
        labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

        predictions_gathered = accelerator.gather(predictions)
        labels_gathered = accelerator.gather(labels)

        true_predictions, true_labels = postprocess(predictions_gathered, labels_gathered)
        metric.add_batch(predictions=true_predictions, references=true_labels)

    results = metric.compute()
    if verbose:
        if accelerator.is_local_main_process:
            print(
                f"epoch {epoch}:",
                {key: results[f"overall_{key}"]
                    for key in ["precision", "recall", "f1", "accuracy"]},)
    interested_metric = results[return_metric]
    return interested_metric



progress_bar = tqdm(range(num_training_steps))


# training loop

all_val_scores = []
best_val_score = 0
test_score = 0
for epoch in range(num_train_epochs):
    # Training
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    val_score = dataloader_evaluation(model, eval_dataloader, return_metric='overall_f1',verbose=True)
    all_val_scores.append(val_score)
    if val_score > best_val_score:
        best_val_score = val_score
        if accelerator.is_local_main_process:
            print(f'>>> Oh~ Current Best Val Score: {best_val_score}')
        # validation上每次有更好的结果，就更新一次test结果
        test_score = dataloader_evaluation(model, test_dataloader, return_metric='overall_f1',verbose=True)
        if DO_SAVE:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(output_dir)
                print('saved!!')
# dataloader_evaluation(model, test_dataloader, return_metric='overall_f1',verbose=True)

        

if accelerator.is_local_main_process:
    print(f'\n>>> Best Validation Score: {best_val_score}')
    print(f'>>> Test Score: {test_score}')

