import argparse
import os
import numpy as np
import datetime
import time
from tqdm import tqdm
import torch
from torch.optim import AdamW
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding
from my_dataset import MyDataloaders
from utils import OrderNamespace, fix_seed
parser = argparse.ArgumentParser(allow_abbrev=False)

# 常用参数：
parser.add_argument('--dataset', type=str, default='bbc_500', help='dataset dir name, in data/')
parser.add_argument('--lang', type=str, default='en', help='en or zh')
parser.add_argument('--train_file', type=str, default='train', help='train filename, name before .csv')
parser.add_argument('--test_file', type=str, default='test', help='test filename, name before .csv')
parser.add_argument('--more_test_files', type=str, default=None, help='test filename, name before .csv, join by ","')
# 其他：
parser.add_argument('--maxlen', type=int, default=512, help='max length of the sequence')
parser.add_argument('--bsz', type=int, default=64, help='batch size')
parser.add_argument('--metric', type=str, default='accuracy', help='metric for early-stopping, "loss" or "accuracy", usually "loss" will train longer')
parser.add_argument('--num_iter', type=int, default=1, help='number of iterations of experiments')
parser.add_argument('--patience', type=int, default=10, help='patience of early-stopping')
parser.add_argument('--comment', type=str, default='', help='extra comment, will be added to the log')
parser.add_argument('--group_head', action='store_true', help='if used, is the first experiment of the group of exps')
# 不重要：
parser.add_argument('--epochs', type=int, default=100, help='max number of epochs')
parser.add_argument('--split_valid_from', type=int, default=None, help='split_valid_from')


# 这里使用parse_known_args以及前面的allow_abbrev=False都是为了防止在交互式中出现问题
args, unknown = parser.parse_known_args(args=None, namespace=OrderNamespace())


args_str = ' '.join([o+'='+str(getattr(args, o)) for o in args.order]) # 按照我指定是顺序排列
log_start_str = '\n==========================\n' if args.group_head else ''
print(args_str)

if not os.path.exists('log/'):
    os.makedirs('log/')
with open('log/%s.txt'%args.dataset, 'a') as f:
    f.write(log_start_str+str(datetime.datetime.now()) + '|' + args.comment + '\n' + args_str + '\n')

# 设置默认的预训练模型：
if args.lang == 'en':
    # checkpoint = 'bert_model/TinyBERT_General_4L_312D'
    checkpoint = 'distilbert-base-uncased'
elif args.lang == 'zh':
    checkpoint = 'bert_model/TinyBERT_4L_zh'
else:
    raise NotImplementedError
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%%
########################################################################
#                            数据集准备
########################################################################
dataset_name = args.dataset  # /data/...
train_filename = args.train_file
test_filename = args.test_file

train_path = 'data/'+dataset_name+'/'+train_filename+'.csv'
test_path = 'data/'+dataset_name+'/'+test_filename+'.csv'
my_dataloaders = MyDataloaders(train_path, test_path, tokenizer, args.maxlen, args.bsz, data_collator, args.split_valid_from)

label2idx, idx2label = my_dataloaders.label2idx, my_dataloaders.idx2label
num_labels=len(my_dataloaders.unique_labels)
# 补充测试集：
if args.more_test_files:
    more_test_dataloaders = []
    from my_dataset import get_dataloader
    more_test_files = args.more_test_files.split(',')
    for file in more_test_files:
        file_path = 'data/'+dataset_name+'/'+file+'.csv'
        more_test_dataloader = get_dataloader(file_path, tokenizer, label2idx, args.maxlen, args.bsz, data_collator)
        more_test_dataloaders.append(more_test_dataloader)


#%%
########################################################################
#                            功能函数、损失函数准备
########################################################################
def init_model():
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)
    return model

num_training_steps = args.epochs * len(my_dataloaders.train_dataloader)  # 注意dataloader是一个batch一个batch输出的
# todo: 这里的lr scheduler似乎不应该直接跟epochs挂钩，不然我这里early-stopping没啥用啊。按道理，应该在优化不动的时候，降低一点lr
# lr_scheduler = get_scheduler('linear', optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

def evaluate(logits, labels):
    preds = torch.argmax(logits, dim=-1).cpu().numpy()
    labels = labels.cpu().numpy()
    return (preds == labels).sum() / len(preds)

def evaluate_from_dataloader(model, dataloader):
    model.eval()
    total, right = 0., 0.
    total_loss = 0.
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits
            loss = outputs.loss  # 这是整个batch的平均loss
        preds = torch.argmax(logits, dim=-1).cpu().numpy()  # 需要先转换到cpu上，才可以转成numpy格式
        labels = batch['labels'].cpu().numpy()
        total += len(preds)
        right += (preds == labels).sum()
        total_loss += loss.item() * len(preds)  # 由于是平均的loss，我想算整个数据集的loss就需要把全部loss都加起来
    acc = right/total
    avg_loss = total_loss/total
    return {'accuracy': acc, 'loss': avg_loss}


#%%
########################################################################
#                            training loop
########################################################################


test_acc_list = []
val_acc_list = []
if not os.path.exists('saved_models/'):
    os.makedirs('saved_models/')
# 跑多个seed然后平均
for i in range(args.num_iter):
    # Fix the random seeds
    fix_seed(i)

    model = init_model()
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)

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
        for batch in tqdm(my_dataloaders.train_dataloader, disable=False):
            batch = {k:v.to(device) for k,v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # validate:
        # note: 由于增强样本是直接混进训练集的，所以划分验证集的时候也会混入增强样本
        res = evaluate_from_dataloader(model, my_dataloaders.val_dataloader)
        val_acc = res['accuracy']
        val_loss = res['loss']
        print('val_acc: %s , val_loss: %s'%(val_acc, val_loss))
        if metric == 'accuracy':
            is_better = val_acc > best_val_acc
        elif metric == 'loss':
            is_better = val_loss < best_val_loss
        if is_better:
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save(model.state_dict(),f=f'saved_models/{args.dataset}.pth')
            print('best model saved!')
            wait = 0
        else:
            wait += 1
            print('current wait:', wait)
            if wait >= args.patience:
                print('early-stop at epoch %s' % epoch)
                break
    val_acc_list.append(best_val_acc)

    # load the best model and evaluate it
    best_model = model
    best_model.load_state_dict(torch.load(f'saved_models/{args.dataset}.pth'))
    test_res = evaluate_from_dataloader(best_model, my_dataloaders.test_dataloader)
    print('test acc: %s, test loss: %s'%(test_res['accuracy'], test_res['loss']))
    t2 = time.time()
    with open('log/%s.txt'%args.dataset, 'a') as f:
        print('iter %s: test acc: %s, test loss: %s | time cost: %.2fs, epochs: %s' %
                (i, test_res['accuracy'], test_res['loss'], t2-t1, epoch), file=f)
    test_acc_list.append(test_res['accuracy'])

    # 补充测试集：
    if args.more_test_files:
        more_test_res_dict = {f: [] for f in more_test_files}
        for file, dataloader in zip(more_test_files, more_test_dataloaders):
            more_test_res = evaluate_from_dataloader(best_model, dataloader)
            more_test_res_dict[file].append(more_test_res['accuracy'])
            print(file, more_test_res['accuracy'])
            with open('log/%s.txt' % args.dataset, 'a') as f:
                print(file, more_test_res['accuracy'], file=f)


# 跑完所有的迭代，总结并记录实验结果
with open('log/%s.txt'%args.dataset, 'a') as f:
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
            avg_test_acc = sum(more_test_res_dict[file])/len(more_test_res_dict[file])
            print('Average %s Acc:'%file, avg_test_acc)
            print('Average %s Acc:'%file, avg_test_acc, file=f)
    print('\n\n', file=f)
