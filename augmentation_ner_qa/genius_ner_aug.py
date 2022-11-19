from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import pipeline
from collections import defaultdict
import random
random.seed(5)
import sys
sys.path.append('../')
from genius_utils import SketchExtractor, List2Dataset


import argparse
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--dataset_name', type=str, default='conll2003', help='dataset name in HF')
parser.add_argument('--train_size', type=int, default=50, help='labeled size')
parser.add_argument('--n_aug', type=int, default=1, help='how many times to augment')
parser.add_argument('--device', type=int, default=0, help='cuda device index, if not found, will switch to cpu')
args = parser.parse_args()


raw_dataset = load_dataset('conll2003')
tag_names = raw_dataset['train'].features['ner_tags'].feature.names
# tag_names: ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

############### util functions: ###############
def get_mention_name(tag):
    # tag: the number/index of the tag name
    # tag_names: the list of tag names
    # mention: ORG, LOC, etc.
    return tag_names[tag].split('-')[-1]


def concat_multiple_sequences(dataset, size=3, overlap=True):
    new_dataset = defaultdict(list)
    l = len(dataset)
    if overlap: # 连续窗口滑动
        for i in range(l-size):
            concat_tokens = np.concatenate(dataset[i:i+size]['tokens'])
            concat_tags = np.concatenate(dataset[i:i+size]['ner_tags'])
            new_dataset['tokens'].append(concat_tokens)
            new_dataset['ner_tags'].append(concat_tags)
    else:  # 互相不重叠
        for i in range(l//size):
            concat_tokens = np.concatenate(dataset[i*size:(i+1)*size]['tokens'])
            concat_tags = np.concatenate(dataset[i*size:(i+1)*size]['ner_tags'])
            new_dataset['tokens'].append(concat_tokens)
            new_dataset['ner_tags'].append(concat_tags)
    return new_dataset


def extract_mentions(tokens, tags):
    """
    return: 
    mentions: []
    mention_dict: {'MISC': [], 'PER': [], 'LOC': [], 'ORG': []}
    """
    mentions = []
    mention_dict = {t:[] for t in list(set([t.split('-')[-1] for t in tag_names])) if t != 'O'}
    for i in range(len(tokens)):
        mention = get_mention_name(tags[i])
        if mention == 'O':
            continue
        if tags[i] % 2 == 1:
            # the start
            mention_dict[mention].append([tokens[i]])
            mentions.append([tokens[i]])
        else:
            # the remaining part
            mention_dict[mention][-1].append(tokens[i])
            mentions[-1].append(tokens[i])
    for k in mention_dict:
        if mention_dict[k]: # not empty
            mention_dict[k] = [' '.join(items) for items in mention_dict[k]]
    mentions = [' '.join(items) for items in mentions]
    return mentions,mention_dict

# entity tagger
class MyTagger:
    def __init__(self, global_mention_dict):
        """
        global_mention_dict: {'PER': [], 'MISC': [], 'ORG': [], 'LOC': []}
        """
        all_mentions = []
        for k in global_mention_dict:
            if k != 'O':
                all_mentions += global_mention_dict[k]
        all_mentions = [tuple(m.split(' ')) for m in all_mentions]
        self.all_mentions = all_mentions
        self.mwe_tokenizer = nltk.tokenize.MWETokenizer(all_mentions,separator=' ')
        self.global_mention_dict = global_mention_dict
        self.reverse_global_mention_dict = {}
        for k in global_mention_dict:
            for e in global_mention_dict[k]:
                self.reverse_global_mention_dict[e] = k

    def tokenize(self, sentence):
        return self.mwe_tokenizer.tokenize(word_tokenize(sentence))
    
    def tag(self, sentence):
        big_words = self.tokenize(sentence) # 包含一些词组
        tags = []
        tokens = []
        for big_word in big_words:
            if big_word in self.reverse_global_mention_dict: # 该词/词组是实体
                full_tag_name = self.reverse_global_mention_dict[big_word]
                for i,single_word in enumerate(word_tokenize(big_word)):
                    if i == 0: # the first word
                        tags.append(tag_names.index('B-'+full_tag_name))
                    else: # the latter word
                        tags.append(tag_names.index('I-'+full_tag_name))
                    tokens.append(single_word)
            else:
                for single_word in word_tokenize(big_word):
                    tags.append(0)
                    tokens.append(single_word)
        assert len(tokens) == len(tags),'.'
        return tokens, tags
#############################################



sketch_extractor = SketchExtractor('yake')

exp_data = raw_dataset['train'].select(range(args.train_size))

longer_data = concat_multiple_sequences(exp_data) # dict_keys(['tokens', 'ner_tags'])

# extract_mentions(longer_data['tokens'][0],longer_data['ner_tags'][0])



# extract all entities to initialize tagger
global_mention_dict = {t:[] for t in list(set([t.split('-')[-1] for t in tag_names])) if t != 'O'}
for tokens, tags in zip(tqdm(longer_data['tokens']), longer_data['ner_tags']):
    mentions, mention_dict = extract_mentions(tokens, tags)
    for k in mention_dict:
        global_mention_dict[k] += mention_dict[k]
        # also add their basic variants
        global_mention_dict[k] += [s.lower() for s in mention_dict[k]]
        global_mention_dict[k] += [s.upper() for s in mention_dict[k]]
        global_mention_dict[k] += [s.title() for s in mention_dict[k]]
        global_mention_dict[k] += [s.capitalize() for s in mention_dict[k]]

for k in global_mention_dict: # remove repetition
    global_mention_dict[k] = list(set(global_mention_dict[k]))
my_tagger = MyTagger(global_mention_dict)




# extracting fragments containing entities
sketches = []
for tokens, tags in zip(tqdm(longer_data['tokens']), longer_data['ner_tags']):
    text = ' '.join(tokens)
    #
    _, kws = sketch_extractor.get_kws(text, max_ngram=3, top=max(len(tokens)//8,5))
    mentions, _ = extract_mentions(tokens, tags)
    sketch = sketch_extractor.get_sketch_from_kws(text, kws+mentions)
    # mentions, _ = extract_mentions(tokens, tags)
    # sketch = sketch_extractor.get_sketch(
    #     text, aspect_keywords=mentions, 
    #     max_ngram=3, top=max(len(tokens)//4,5)
    #     )
    sketches.append(sketch)

sketch_dataset = List2Dataset(sketches)

genius = pipeline('text2text-generation',model='beyond/genius-base',device=0)


## genius generating
long_augmented_dataset = defaultdict(list)
short_augmented_dataset = defaultdict(list)

for i in range(args.n_aug):
    for out in tqdm(genius(sketch_dataset, num_beams=3, do_sample=True, max_length=100, batch_size=32)):
        generated_text = out[0]['generated_text']
        # print(f'  >>> generated_text: {generated_text}')
        # tag the generated sentence
        new_tokens, new_tags = my_tagger.tag(generated_text)
        long_augmented_dataset['tokens'].append(new_tokens)
        long_augmented_dataset['ner_tags'].append(new_tags)
        # shorter sequences:
        sents = sent_tokenize(generated_text)
        for sent in sents:
            if len(sent.split(' ')) <= 3:
                continue
            new_tokens, new_tags = my_tagger.tag(sent)
            short_augmented_dataset['tokens'].append(new_tokens)
            short_augmented_dataset['ner_tags'].append(new_tags)
print(f'>>> Num of long generated examples: {len(long_augmented_dataset["tokens"])}')
print(f'>>> Num of short generated examples: {len(short_augmented_dataset["tokens"])}')


# Simple Filtering: 
print(f'filtering too short seqs...')
tokens_list = []
tags_list = []
for tokens,tags in zip(short_augmented_dataset['tokens'], short_augmented_dataset['ner_tags']):
    if len(list(set(tags))) > 1 and len(tags) > 5: 
        tokens_list.append(tokens)
        tags_list.append(tags)

short_augmented_dataset = {'tokens':tokens_list, 'ner_tags':tags_list}
print(f'>>> Num of filtered short examples: {len(short_augmented_dataset["tokens"])}')


df = pd.DataFrame(long_augmented_dataset)
file_name = f'../ner_data/{args.dataset_name}-{args.train_size}-genius-long-naug-{args.n_aug}.pkl'
df.to_pickle(file_name)  # don't save csv, or the list will turn into string
print(file_name)
df = pd.DataFrame(short_augmented_dataset)
file_name = f'../ner_data/{args.dataset_name}-{args.train_size}-genius-short-naug-{args.n_aug}.pkl'
df.to_pickle(file_name)  
print(file_name)