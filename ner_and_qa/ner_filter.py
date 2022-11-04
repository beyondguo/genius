from lib2to3.pgen2 import token
from cv2 import threshold
from transformers import pipeline
import pandas as pd
from datasets import Dataset as HFDataset
import torch

################################################################
model_checkpoint = "saved_models/ner/base-500"
aug_file_name = 'ner_data/conll03-500-K2T-frame-aug1_short_v5.pkl'
################################################################


token_classifier = pipeline(
    "token-classification", model=model_checkpoint, aggregation_strategy="simple"
)

label_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']


df = pd.read_pickle(aug_file_name)
augmented_dataset = HFDataset.from_pandas(df)


# augmented_dataset = {'tokens':augmented_dataset['tokens'],'ner_tags':augmented_dataset['ner_tags']}
# # 对太长的序列进行切分，可反复调用直到切不动
# def cut_too_long_sequences(tokens_list, tags_list):
#     short_tokens_list = []
#     short_tags_list = []
#     for tokens, tags in zip(tokens_list, tags_list):
#         if '.' in tokens and len(tokens) > 10:
#             index = tokens.index('.')
#             if index < len(tokens) - 1 and index > 1:
#                 short_tokens_list += [tokens[:index+1], tokens[index+1:]]
#                 short_tags_list += [tags[:index+1], tags[index+1:]]
#             else:
#                 short_tokens_list.append(tokens)
#                 short_tags_list.append(tags)
#         else:
#             short_tokens_list.append(tokens)
#             short_tags_list.append(tags)
#     assert len(short_tokens_list) == len(short_tags_list), 'not good~'
#     return short_tokens_list, short_tags_list
# # Cutting: 把生成的长序列切短
# print(f'cutting long seqs...')
# # cutting some too-long generated sequences
# augmented_dataset['tokens'], augmented_dataset['ner_tags'] = \
# cut_too_long_sequences(*cut_too_long_sequences(augmented_dataset['tokens'],augmented_dataset['ner_tags']))
# # do it again:
# augmented_dataset['tokens'], augmented_dataset['ner_tags'] = \
# cut_too_long_sequences(*cut_too_long_sequences(augmented_dataset['tokens'],augmented_dataset['ner_tags']))
# print(f'>>> Num of generated training examples: {len(augmented_dataset["tokens"])}')

# # Simple Filtering: 并进行一些简单过滤
# print(f'filtering too short seqs...')
# tokens_list = []
# tags_list = []
# for tokens,tags in zip(augmented_dataset['tokens'], augmented_dataset['ner_tags']):
#     if len(list(set(tags))) > 1 and len(tags) > 5: # 只保留至少一个实体的,且词数大于5
#         tokens_list.append(tokens)
#         tags_list.append(tags)

# augmented_dataset['tokens'], augmented_dataset['ner_tags'] = tokens_list, tags_list
# print(f'>>> Num of generated training examples: {len(augmented_dataset["tokens"])}')

tokenizer = token_classifier.tokenizer
model = token_classifier.model

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
        examples["tokens"], truncation=True, is_split_into_words=True, return_tensors='pt'
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = torch.LongTensor(new_labels)
    return tokenized_inputs

from tqdm import tqdm
good_idxs = []
accs = []

# augmented_dataset = HFDataset.from_pandas(pd.DataFrame(augmented_dataset))
for i in tqdm(range(len(augmented_dataset))):
    out = tokenize_and_align_labels(augmented_dataset[i:i+1])
    pred = model(**out).logits[0]

    true_labels = out.labels.tolist()[0][1:-1]  # 去掉首尾
    pred_labels = torch.argmax(pred, dim=1).tolist()[1:-1]
    acc = sum([a==b for a,b in zip(true_labels, pred_labels)])/len(true_labels)
    print(acc)
    accs.append(acc)
    # if acc >= threshold:
    #     good_idxs.append(i)
    #     print('nice')


threshold = 1
good_idxs = [i for i in range(len(accs)) if accs[i]>=threshold]
print(len(good_idxs))


# def get_mention_name(tag):
#     # tag: the number/index of the tag name
#     # tag_names: the list of tag names
#     # mention: ORG, LOC, etc.
#     return label_names[tag].split('-')[-1]

# # 单独把实体抽出来
# def extract_mentions(tokens, tags):
#     """
#     return: 
#     mentions: []
#     mention_dict: {'MISC': [], 'PER': [], 'LOC': [], 'ORG': []}
#     """
#     mentions = []
#     mention_dict = {t:[] for t in list(set([t.split('-')[-1] for t in label_names])) if t != 'O'}
#     for i in range(len(tokens)):
#         mention = get_mention_name(tags[i])
#         if mention == 'O':
#             continue
#         if tags[i] % 2 == 1:
#             # the start
#             mention_dict[mention].append([tokens[i]])
#             mentions.append([tokens[i]])
#         else:
#             # the remaining part
#             mention_dict[mention][-1].append(tokens[i])
#             mentions[-1].append(tokens[i])
#     for k in mention_dict:
#         if mention_dict[k]: # not empty
#             mention_dict[k] = [' '.join(items) for items in mention_dict[k]]
#     mentions = [' '.join(items) for items in mentions]
#     return mentions,mention_dict
# i = 481
# s = ' '.join(augmented_dataset[i]['tokens'])
# mentions,_ = extract_mentions(augmented_dataset[i]['tokens'],augmented_dataset[i]['ner_tags'])
# res = [item['word'] for item in token_classifier(s)]
# print(s)
# print('>>> orig: ', mentions)
# print('>>> pred: ', res)



# good_idxs = [1, 2, 6, 7, 11, 12, 14, 17, 18, 20, 21, 23, 24, 25, 26, 32, 37, 41, 42, 43, 44, 45, 46, 47, 53, 59, 64, 88, 89, 90, 103, 122, 144, 147, 148, 170, 197, 199, 214, 227, 232, 240, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 256, 257, 259, 260, 261, 269, 272, 283, 288, 289, 290, 291, 292, 294, 301, 305, 306, 307, 309, 313, 314, 315, 316, 317, 321, 322, 343, 371, 372, 374, 384, 385, 397, 400, 408, 409, 421, 451, 475, 477, 484, 485, 489, 492]
from collections import defaultdict
new_augmented_dataset = defaultdict(list)
for idx in good_idxs:
    new_augmented_dataset['tokens'].append(augmented_dataset['tokens'][idx])
    new_augmented_dataset['ner_tags'].append(augmented_dataset['ner_tags'][idx])
print(len(new_augmented_dataset['tokens']))






# 保存数据集
new_df = pd.DataFrame(new_augmented_dataset)
new_file_name = aug_file_name.split('.')[0]+'_filtered_%s'%threshold+'.pkl'
new_df.to_pickle(new_file_name)  # 不能保存csv，因为涉及到保存list，csv会变成string了
print(new_file_name)

