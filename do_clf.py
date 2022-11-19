
import os
import time
import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
import torch.nn.functional as F
from collections import defaultdict
from utils import OrderNamespace, fix_seed, get_dataloader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
import argparse
parser = argparse.ArgumentParser(allow_abbrev=False)

# main params:
parser.add_argument('--dataset', type=str, default='bbc_500', help='dataset dir name, in data/')
parser.add_argument('--checkpoint', type=str, default='distilbert-base-cased', help='classification model checkpoint')
parser.add_argument('--train_file', type=str, default='train', help='train filename, name before .csv')
parser.add_argument('--dev_file', type=str, default='dev', help='dev filename, name before .csv')
parser.add_argument('--test_file', type=str, default='test', help='test filename, name before .csv')
parser.add_argument('--more_test_files', type=str, default=None, help='test filename, name before .csv, join by ","')
# other params:
parser.add_argument('--maxlen', type=int, default=512, help='max length of the sequence')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
parser.add_argument('--train_bz', type=int, default=32, help='train batch size')
parser.add_argument('--eval_bz', type=int, default=32, help='evaluation batch size')
parser.add_argument('--metric', type=str, default='loss', help='metric for early-stopping, "loss" or "accuracy", usually "loss" will train longer')
parser.add_argument('--num_iter', type=int, default=1, help='number of iterations of experiments')
parser.add_argument('--patience', type=int, default=3, help='patience of early-stopping')
parser.add_argument('--no_early_stop', action='store_true', default=False, help='if set, will NOT use early-stopping')
parser.add_argument('--comment', type=str, default='', help='extra comment, will be added to the log')
parser.add_argument('--group_head', action='store_true', help='if used, is the first experiment of the group of exps')
parser.add_argument('--epochs', type=int, default=100, help='max number of epochs')


# parse_known_args, allow_abbrev=False are to prevent potential problems in interactive mode
args, unknown = parser.parse_known_args(args=None, namespace=OrderNamespace())


args_str = ' '.join([o+'='+str(getattr(args, o)) for o in args.order]) # ordering in specified order
log_start_str = '\n==========================\n' if args.group_head else ''
print(args_str)

log_dir_name = 'clf-log'
if not os.path.exists(f'./{log_dir_name}/'):
    os.makedirs(f'./{log_dir_name}/')
with open(f'{log_dir_name}/{args.dataset}.txt', 'a') as f:
    f.write(log_start_str+str(datetime.datetime.now()) + '|' + args.comment + '\n' + args_str + '\n')


checkpoint = args.checkpoint
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


########################################################################
#                            dataset
########################################################################

train_path = 'data_clf/'+args.dataset+'/'+ args.train_file+'.csv'
dev_path = 'data_clf/'+args.dataset+'/'+args.dev_file+'.csv'
test_path = 'data_clf/'+args.dataset+'/'+args.test_file+'.csv'

train_df = pd.read_csv(train_path)  
train_labels = list(train_df['label'])
unique_labels = sorted(list(set(train_labels)))
label2idx = {unique_labels[i]: i for i in range(len(unique_labels))}
idx2label = {label2idx[label]: label for label in label2idx}

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_dataloader = get_dataloader(train_path, tokenizer, label2idx, args.maxlen, args.train_bz, data_collator)
dev_dataloader = get_dataloader(dev_path, tokenizer, label2idx, args.maxlen, args.eval_bz, data_collator)
test_dataloader = get_dataloader(test_path, tokenizer, label2idx, args.maxlen, args.train_bz, data_collator)

# addtional test sets：
if args.more_test_files:
    more_test_dataloaders = []
    more_test_files = args.more_test_files.split(',')
    for file in more_test_files:
        file_path = 'data_clf/'+args.dataset+'/'+file+'.csv'
        more_test_dataloader = get_dataloader(file_path, tokenizer, label2idx, args.maxlen, args.eval_bz, data_collator)
        more_test_dataloaders.append(more_test_dataloader)


########################################################################
#                       util functions
########################################################################
def init_model():
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=len(unique_labels))
    return model

def evaluate(logits, labels):
    preds = torch.argmax(logits, dim=-1).cpu().numpy()
    labels = labels.cpu().numpy()
    return (preds == labels).sum() / len(preds)

def evaluate_from_dataloader(model, dataloader, disable_tqdm=True):
    model.eval()
    total, right = 0., 0.
    total_loss = 0.
    for batch in tqdm(dataloader, disable=disable_tqdm):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits
            loss = outputs.loss  # average loss of the whole batch
        preds = torch.argmax(logits, dim=-1).cpu().numpy()  # first to cpu, then to numpy format
        labels = batch['labels'].cpu().numpy()
        total += len(preds)
        right += (preds == labels).sum()
        total_loss += loss.item() * len(preds)
    acc = right/total
    avg_loss = total_loss/total
    return {'accuracy': acc, 'loss': avg_loss}


########################################################################
#                        training loop
########################################################################


test_acc_list = []
val_acc_list = []
more_test_res_dict = defaultdict(list)
if not os.path.exists('saved_models/'):
    os.makedirs('saved_models/')
# average of multiple seeds
for i in range(args.num_iter):
    # Fix the random seeds
    fix_seed(i)

    model = init_model()
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    t1 = time.time()
    best_val_acc = 0.
    best_val_loss = float('inf')
    wait = 0
    is_better = False
    metric = args.metric  # 'accuracy' or 'loss'
    for epoch in range(args.epochs):
        print('====epoch %s===='%epoch)
        # train:
        model.train()
        for batch in tqdm(train_dataloader, disable=False):
            batch = {k:v.to(device) for k,v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # validate with early-stopping:
        res = evaluate_from_dataloader(model, dev_dataloader)
        val_acc = res['accuracy']
        val_loss = res['loss']
        print('val_acc: %s , val_loss: %s'%(val_acc, val_loss))
        if not args.no_early_stop:
            if metric == 'accuracy':
                is_better = val_acc > best_val_acc  
            elif metric == 'loss':
                is_better = val_loss < best_val_loss
            if is_better:
                best_val_acc = val_acc
                best_val_loss = val_loss
                torch.save(model.state_dict(),f=f'saved_models/{args.dataset}_{args.checkpoint}_{args.train_file}.pkl',_use_new_zipfile_serialization=False)
                print('best model saved!')
                wait = 0
            else:
                wait += 1
                print('current wait:', wait)
                if wait >= args.patience:
                    print('early-stop at epoch %s' % epoch)
                    break
    if args.no_early_stop:
        torch.save(model.state_dict(),f=f'saved_models/{args.dataset}_{args.checkpoint}_{args.train_file}.pkl',_use_new_zipfile_serialization=False)
        print('last model saved!')
        best_val_acc = val_loss # last val

    val_acc_list.append(best_val_acc)

    # load the best model and test it
    model.load_state_dict(torch.load(f'saved_models/{args.dataset}_{args.checkpoint}_{args.train_file}.pkl'))
    test_res = evaluate_from_dataloader(model, test_dataloader, disable_tqdm=False)
    print('test acc: %s, test loss: %s'%(test_res['accuracy'], test_res['loss']))
    t2 = time.time()
    with open(f'{log_dir_name}/{args.dataset}.txt', 'a') as f:
        print('iter %s: test acc: %s, test loss: %s | time cost: %.2fs, epochs: %s' %
                (i, test_res['accuracy'], test_res['loss'], t2-t1, epoch), file=f)
    test_acc_list.append(test_res['accuracy'])

    # addtionla test:
    if args.more_test_files:
        for file, dataloader in zip(more_test_files, more_test_dataloaders):
            more_test_res = evaluate_from_dataloader(model, dataloader, disable_tqdm=False)
            more_test_res_dict[file].append(more_test_res['accuracy'])
            print(file, more_test_res['accuracy'])
            with open(f'{log_dir_name}/{args.dataset}.txt', 'a') as f:
                print(file, more_test_res['accuracy'], file=f)


# recording the results
with open(f'{log_dir_name}/{args.dataset}.txt', 'a') as f:
    avg_val_acc = sum(val_acc_list) / len(val_acc_list)
    val_acc_std = np.std(val_acc_list)
    avg_test_acc = sum(test_acc_list)/len(test_acc_list)
    test_acc_std = np.std(test_acc_list)
    print('----------',file=f)
    print('Average Val Acc:', avg_val_acc)
    print('Average Val Acc:', avg_val_acc, file=f)
    print('Val std:',val_acc_std, file=f)
    print('Average Test Acc:', avg_test_acc)
    print('Average Test Acc:', avg_test_acc, file=f)
    print('Test std:', test_acc_std, file=f)

    if args.more_test_files:
        for file in more_test_files:
            print(more_test_res_dict[file])
            avg_test_acc = sum(more_test_res_dict[file])/len(more_test_res_dict[file])
            print('Average %s Acc:'%file, avg_test_acc)
            print('Average %s Acc:'%file, avg_test_acc, file=f)
    print('\n\n', file=f)
