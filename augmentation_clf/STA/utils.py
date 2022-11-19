import argparse
import torch
import numpy as np
import random
import os

def makedir(path):
    is_exist = os.path.exists(path)
    if is_exist:
        return '%s already exists!'%path
    else:
        os.makedirs(path)

def setup_seed(seed):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


class OrderNamespace(argparse.Namespace):
    def __init__(self, **kwargs):
        self.__dict__['order'] = []
        super(OrderNamespace, self).__init__(**kwargs)
    def __setattr__(self,attr,value):
        #  如果没有这个if，args中的str类型会在log中打印两次。
        #  猜测可能是由于父类会对str重复调用__setattr__方法的缘故
        if attr not in self.__dict__['order']:
            self.__dict__['order'].append(attr)
        super(OrderNamespace, self).__setattr__(attr, value)


def fix_seed(i):
    random.seed(i)
    np.random.seed(i)
    torch.manual_seed(i)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(i)


## for cnn:
from collections import Counter
import gensim

def build_vocab(content_word_list, vocab_size):
    print('begin building vocabulary')
    all_word_list = []
    for word_list in content_word_list:
        all_word_list.extend(word_list)

    counter = Counter(all_word_list)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))

    word_to_id = dict(zip(words, range(1, len(words) + 1)))
    word_to_id['<unk>'] = 0
    # id_to_word = {id_: word for word, id_ in word_to_id.items()}
    return word_to_id


def text2encoding(words, word_to_id, maxlen, padding=False):
    encoding = [word_to_id[word] for word in words if word in word_to_id]
    if padding:
        l = len(encoding)
        encoding = encoding[:maxlen] if l >= maxlen else encoding + [0] * (maxlen - l)
    return encoding


def get_word2vec_weight(lang, word_to_id):
    vocab_size = len(word_to_id)
    print('begin loading word2vec model')
    if lang == 'en':
        word2vec_name = 'GoogleNews-vectors-negative300.bin'  # 词向量模型文件
        # word2vec_path = os.path.join(os.path.dirname(__file__),'weights', word2vec_name)
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format('weights/%s'%word2vec_name, binary=True)
    elif lang == 'zh':
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format("weights/synonyms_words.vector",
                                                           binary=True, unicode_errors='ignore')
    else:
        return "only support en or zh"
    word_embedding_dim = w2v_model[0].shape[0]
    # weight = torch.zeros(vocab_size, word_embedding_dim)
    weight = np.zeros((vocab_size, word_embedding_dim), dtype='float64')

    for word in w2v_model.index_to_key:  # 新版的gensim，通过这个index_to_key得到vocab的list
        if word in word_to_id:
            weight[word_to_id[word], :] = w2v_model[word]
    weight = torch.from_numpy(weight)
    return weight, word_embedding_dim