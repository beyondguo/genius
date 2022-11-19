"""
「角色关键词提取/ Role Keywords Extraction」
目前的用法感觉还比较尴尬，分两步进行：
1. 先得把全部的训练数据和标签丢进来，计算相似度和统计信息
2. 再根据利用这些统计信息，来提取关键词
所以，它是一个静态的算法，如果数据集扩增了，就需要重新统计信息。
这就跟textrank等即时算法不同。相当于这个是一个**全局**的关键词提取算法。
"""
from cProfile import label
from gensim.models.keyedvectors import KeyedVectors
import math
import numpy as np
import jieba
import nltk
import pickle
from tqdm import tqdm
import warnings
import os

warnings.filterwarnings('ignore')


# compute tf，idf value
def get_tf(t, document_words):
    """
    t: term
    document_words: split document words
    term must be in document words
    """
    return document_words.count(t) / len(document_words)


def get_idf(t, ds):
    """
    ds: dataset, document list
    """
    n = len(ds)
    df = len([d for d in ds if t in d])
    return math.log(n / (1 + df))


def get_wllr(in_class_freq, out_class_freq):
    """
    WLLR: weighted log likelihood ratio
    r(w,y) = p(w|y)*log(p(w|y)/p(w|y^))
    WLLR reflects the given word's correlation with a class.
    The same word in a class will always has same WLLR value,
    no matter in which sample.
    """
    wllr = in_class_freq * math.log10(in_class_freq / out_class_freq)
    return wllr

def get_median(scores):
    scores = sorted(scores, reverse=True)
    l = len(scores)
    if l % 2 == 0:
        return (scores[int(l/2-1)]+scores[int(l/2)])/2
    else:
        return scores[int((l-1)/2)]

def get_quartiles(scores):
    """
    获取分位数。
    Q1：上四分位数；
    Q2：中位数；
    Q3：下四分位数。
    """
    try:
        scores = sorted(scores, reverse=True)
        if len(scores) == 1:
            return {'Q1':scores[0],'Q2':scores[0],'Q3':scores[0]}
        if len(scores) == 2:
            return {'Q1':(scores[0]+(scores[0]+scores[1])/2)/2, 'Q2':(scores[0]+scores[1])/2, 'Q3':(scores[1]+(scores[0]+scores[1])/2)/2}

        l = len(scores)
        if l % 2 == 0:
            return {'Q1':get_median(scores[:int(l/2)]), 'Q2': get_median(scores), 'Q3': get_median(scores[int(l/2):])}
        else:
            return {'Q1':get_median(scores[:int((l-1)/2)]), 'Q2': get_median(scores), 'Q3': get_median(scores[int((l+1)/2):])}
    except Exception as e:
        print(e)
        print('scores',scores)

def normalize(score, Min, Max): # min-max normalization
    if score == Min:
        return 1e-5
    return (score-Min)/(Max-Min)

####################################################################################
####################################################################################
class KeywordsExtractor:
    def __init__(self, lang):
        assert lang in ['zh', 'en'], "only support 'zh'(for Chinese) or 'en'(for English)"
        language = 'English' if lang == 'en' else 'Chinese'
        print(f'Language: {language}')
        self.lang = lang
        self.stop_words = []
        if lang == 'en':
            print('Loading word vectors......')
            # 这里使用的是GoogleNews词向量，个人感觉比Glove词向量好
            # tips: If use Glove, set no_header=True, since Glove file misses a header
            self.w2v_model = KeyedVectors.load_word2vec_format(os.path.join(os.path.dirname(__file__), 'weights', 'GoogleNews-vectors-negative300.bin'),
                                                               binary=True, unicode_errors='ignore')
            f = open(os.path.join(os.path.dirname(__file__), 'stopwords', 'en_stopwords.txt'), encoding='utf8')
            for stop_word in f.readlines():
                self.stop_words.append(stop_word[:-1])
            self.tokenizer = nltk.tokenize.word_tokenize
        if lang == 'zh':
            print('Loading word vectors......')
            # 这里是下载了synonyms包中自带的词向量文件，包含40w+的词汇
            # 可替换成其他词向量文件
            self.w2v_model = KeyedVectors.load_word2vec_format(os.path.join(os.path.dirname(__file__), 'weights', 'synonyms_words.vector'),
                                                               binary=True, unicode_errors='ignore')
            # 中文停用词，使用哈工大停用词表：
            f = open(os.path.join(os.path.dirname(__file__), 'stopwords', 'zh_stopwords.txt'), encoding='utf8')
            for stop_word in f.readlines():
                self.stop_words.append(stop_word[:-1])
            self.tokenizer = jieba.lcut
        self.vocab = self.w2v_model.index_to_key

    def get_text_vec(self, s):
        """
        获取一段文本（一个字，一个词，一个短语，一段话都可以）的w2v向量表示
        get the w2v representation of a given string
        - 如果这个文本直接在词汇表中了，就直接查询w2v
        - 如果不在，先通过jieba分词得到更小的单位，再依次查询这些小单位的w2v，平均得到整体的表示
        - 如果jieba分词之后还不在词库中，那就按照字符级别进行分割
        # future:
        最近看了一篇论文，可以根据每个词跟类别之间的关系设置一个权重，从而改善句子的表示。后面可以试试。
        """
        if s in self.vocab:
            v = self.w2v_model[s]
        else:
            # the就是随便找了一个词汇表中肯定有的词，为了获取一个同样形状的空向量
            v = np.zeros_like(self.w2v_model['the'])
            count = 0
            words = []
            # 先确定哪些成分拥有词向量
            for w in self.tokenizer(s):  # 先进行分词，得到词粒度
                # 在词库中
                if w in self.vocab:
                    words.append(w)
                # 分词之后还不在词库中，那就使用最小的字粒度，仅针对中文
                elif w not in self.vocab and self.lang == 'zh':
                    for c in w:
                        if c in self.vocab:
                            words.append(c)
            # 对这些词的词向量进行平均
            for t in words:
                if t in self.vocab:
                    v += self.w2v_model[t]
                    count += 1
            if count:
                v /= count
            else:
                v = self.w2v_model['the']
                # print('use random embedding for', s)
        return v

    def compute_similarity_by_vector(self, v1, v2):
        return self.w2v_model.cosine_similarities(v1, v2.reshape(1, -1))[0]

    def compute_similarity_by_text(self, s1, s2):
        """
        计算两段文本（词，句，段）的向量余弦相似度
        compute the cosine similarity of two pieces of text
        """
        vec_pair = [self.get_text_vec(s1), self.get_text_vec(s2)]
        # 这里直接使用gensim的自带函数：
        return self.w2v_model.cosine_similarities(vec_pair[0], vec_pair[1].reshape(1, -1))[0]

    #######################################################################
    # label similarity & label relation是跟具体样本无关的，所以可以先统一计算并保存
    def compute_label_similarity(self, contents, labels, label_desc_dict=None, num_words=None):
        """
        给定一批标注的文本，按类别计算每个词跟对应类别的语义相似度
        num_words: 每个样本最大词数
        """
        assert len(contents) == len(labels), 'must be same length!'
        global_ls_dict = {}
        sorted_ls_dict = {}
        words_dict = {label: [] for label in set(labels)}
        print('tokenizing....')
        # 每个类的词单独存放，减少无用计算（在该类中没有出现的词，在后面用不到）
        for content, label in zip(tqdm(contents), labels):
            if num_words:
                words_dict[label] += self.tokenizer(content)[:num_words]
            else:
                words_dict[label] += self.tokenizer(content)
        label_desc_vec = {}
        for label in set(labels):
            global_ls_dict[label] = {}
            print('computing Label-Similarity for label:', label)
            for w in tqdm(list(set(words_dict[label]))):
                if label_desc_dict is None:
                    global_ls_dict[label][w] = self.compute_similarity_by_text(w, label)
                else:
                    # 这一段主要是为了优化计算速度，避免重复计算label的向量表示
                    # 所以先计算好label description的向量，然后直接算相似度。
                    if label in label_desc_vec:  # 先查是否计算过
                        label_v = label_desc_vec[label]
                    else:
                        try:
                            label_v = self.get_text_vec(label_desc_dict[int(label)])
                        except:
                            label_v = self.get_text_vec(label_desc_dict[str(label)])
                        label_desc_vec[label] = label_v
                    word_v = self.get_text_vec(w)
                    global_ls_dict[label][w] = self.compute_similarity_by_vector(word_v, label_v)

            sorted_ls_dict[label] = [(pair[0], pair[1]) for pair in sorted(global_ls_dict[label].items(),
                                                                           key=lambda kv: kv[1], reverse=True)]
        return global_ls_dict, sorted_ls_dict

    def compute_label_correlation(self, contents, labels, num_words=None):
        """
        给定一批标注的文本，按类别计算每个词跟对应类别的相关性.
        global_doc_count记录每个类中每个词出现在多少个文章里
        对于计算wllr来说，某个词w对于某个类l，
        其in-class-count = gdc[l][w]，
        out-class-count = sum_i(gdc[other_l_i][w])
        则 in_class_freq = in_class_count/gdc[l]['total_count']
        num_words: 每个样本最大词数
        """
        assert len(contents) == len(labels), 'must be same length!'
        global_doc_count = {}
        global_wllr_dict = {}
        sorted_wllr_dict = {}
        print('counting #doc for each word...')
        for label in set(labels):
            global_doc_count[label] = {}  # 每个dict，key为word，value为所在文章数
            global_wllr_dict[label] = {}
            global_doc_count[label]['TOTAL_COUNT'] = 0  # 记录该类别共有多少文章
        for content, label in zip(tqdm(contents), labels):
            if num_words:
                words = self.tokenizer(content)[:num_words]
            else:
                words = self.tokenizer(content)
            global_doc_count[label]['TOTAL_COUNT'] += 1
            for w in set(words):
                if w in global_doc_count[label]:  # w already in this dict
                    global_doc_count[label][w] += 1
                else:  # w not in the dict
                    global_doc_count[label][w] = 1
        # 对每个label，计算所有词的wllr值：
        for label in set(labels):
            print('computing Label-Correlation for label:', label)
            num_in_class_docs = global_doc_count[label]['TOTAL_COUNT']
            num_out_class_docs = sum([global_doc_count[l]['TOTAL_COUNT'] for l in set(labels) if l != label])
            assert num_in_class_docs + num_out_class_docs == len(labels), 'hey bro, check here!'
            for w in tqdm(list(global_doc_count[label].keys())):
                if w == 'TOTAL_COUNT':
                    continue
                in_count = global_doc_count[label][w]
                out_count = max(sum([global_doc_count[l].get(w,0) for l in set(labels) if l != label]), 1e-5)
                in_class_freq = in_count / num_in_class_docs
                out_class_freq = out_count / num_out_class_docs
                global_wllr_dict[label][w] = get_wllr(in_class_freq, out_class_freq)
            # 排个序
            sorted_wllr_dict[label] = [(pair[0], pair[1]) for pair in sorted(global_wllr_dict[label].items(),
                                                                             key=lambda kv: kv[1], reverse=True)]
        return global_wllr_dict, sorted_wllr_dict

    ########################################################################################################
    ########################################################################################################
    def extract_global_role_kws(self, labels, sorted_ls_dict, sorted_wllr_dict):
        global_class_indicating_words = {}
        global_fake_class_indicating_words = {}
        for label in set(labels):
            # label-correlation排在前30%的词
            high_lr = [w[0] for w in sorted_wllr_dict[label][:int(0.2 * len(sorted_wllr_dict[label]))]]
            # 相似度在前30%以内，那就是比较好的类别指示词了：
            high_ls = [w[0] for w in sorted_ls_dict[label][:int(0.3 * len(sorted_ls_dict[label]))]]
            global_class_indicating_words[label] = [w for w in high_lr if w in high_ls]
            # 相似度却排在80%之后，那就是比较强的noise了：
            high_ls = [w[0] for w in sorted_ls_dict[label][:int(0.8 * len(sorted_ls_dict[label]))]]
            global_fake_class_indicating_words[label] = [w for w in high_lr if w not in high_ls]
        return global_class_indicating_words, global_fake_class_indicating_words


    def global_role_kws_extraction_one_line(self, contents, labels, label_desc_dict=None, num_words=None, 
                                            output_dir='.', name='', overwrite=False):
        ls_save_path = f'{output_dir}/global_ls_dict_{name}.pkl'
        lr_save_path = f'{output_dir}/global_lr_dict_{name}.pkl'
        global_roles_save_path = f'{output_dir}/global_kws_dict_{name}.pkl'
        
        # ls
        if os.path.exists(ls_save_path) and not overwrite: # 保存过，就直接加载
            print('ls dict already exists at: ',ls_save_path)
            with open(ls_save_path, 'rb') as f:
                global_ls_dict = pickle.load(f)
        else: # 否则现场计算，并保存
            global_ls_dict, sorted_ls_dict = self.compute_label_similarity(contents,labels,label_desc_dict,num_words)
            with open(f'{output_dir}/global_ls_dict_{name}.pkl','wb') as f:
                pickle.dump(global_ls_dict, f)
                print('saved at',f'{output_dir}/global_ls_dict_{name}.pkl')
        
        # lr
        if os.path.exists(lr_save_path) and not overwrite:
            print('lr dict already exists at: ',lr_save_path)
            with open(lr_save_path, 'rb') as f:
                global_lr_dict = pickle.load(f)
        else:
            global_lr_dict, sorted_lr_dict = self.compute_label_correlation(contents,labels,num_words)
            with open(f'{output_dir}/global_lr_dict_{name}.pkl','wb') as f:
                pickle.dump(global_lr_dict, f)
                print('saved at',f'{output_dir}/global_lr_dict_{name}.pkl')
        
        # then, global role keywords
        if os.path.exists(global_roles_save_path) and not overwrite:
            print('global roles dict already exists at: ',global_roles_save_path)
            with open(global_roles_save_path, 'rb') as f:
                global_roles_dict = pickle.load(f)
        else:
            global_roles_dict = global_role_kws_extraction(global_lr_dict, global_ls_dict, list(set(labels)))
            print('First level keys: ',list(global_roles_dict.keys()))
            print('Second level keys: ',list(global_roles_dict[list(global_roles_dict.keys())[0]].keys()))
            with open(f'{output_dir}/global_kws_dict_{name}.pkl','wb') as f:
                pickle.dump(global_roles_dict, f)
                print('already saved at',f'{output_dir}/global_kws_dict_{name}.pkl')
        return {'global_ls':global_ls_dict,
                'global_lr':global_lr_dict,
                'global_roles':global_roles_dict}


# -------------------------------------------------------------------------------------------

def role_kws_extraction_single(words, label, global_ls_dict, global_lr_dict, bar='Q2',skip_words=[]):
    """
    通过分位数这种大家好接受的标准来划分集合，然后归一化，然后排序。目前来看效果不错。
    根据得分的分位数进行划分高低的标准，可选择Q1，Q2，Q3三种分位数。
    通用类别指示词（CCW）：高lr跟高ls的交集
    特殊类别指示词（SCW）：低lr跟高ls的交集
    噪音类别指示词（FCW）：高lr跟低ls的交集
    类别无关词（IW）：低lr跟低ls的交集
    划分四种类别之后，每个集合先进行数值归一化，然后进行乘积或者除法的排序。
    或者先简单点儿，每个集合内部，按看lr高的，就按照lr排序，lr低的就按照-lr排序
    """
    lr_dict = {}
    ls_dict = {}
    for w in set(words):
        # filter punctuations and stop words
        if w in skip_words:
            continue
        ls_dict[w] = global_ls_dict[label][w]
        lr_dict[w] = global_lr_dict[label][w]
    if len(ls_dict) == 0:  # 被过滤地一个词都不剩了，就不过滤重新来一次
        for w in set(words):
            ls_dict[w] = global_ls_dict[label][w]  # 使用exp()，防止负值的影响
            lr_dict[w] = global_lr_dict[label][w]
    lr_bar = get_quartiles(list(lr_dict.values()))[bar]
    ls_bar = get_quartiles(list(ls_dict.values()))[bar]
    lr_min, lr_max = min(list(lr_dict.values())), max(list(lr_dict.values()))
    ls_min, ls_max = min(list(ls_dict.values())), max(list(ls_dict.values()))
    words_lr_sorted = [p[0] for p in sorted(lr_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)]
    words_ls_sorted = [p[0] for p in sorted(ls_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)]

    ccw_dict, scw_dict, fcw_dict, iw_dict = {}, {}, {}, {}
    for w in ls_dict:  # todo：是乘除法好，还是加减法好？
        if lr_dict[w] >= lr_bar and ls_dict[w] >= ls_bar: # ccw
            ccw_dict[w] = normalize(lr_dict[w],lr_min,lr_max) * normalize(ls_dict[w],ls_min,ls_max)
        if lr_dict[w] < lr_bar and ls_dict[w] >= ls_bar: # scw
            scw_dict[w] = normalize(ls_dict[w],ls_min,ls_max) / normalize(lr_dict[w],lr_min,lr_max)
        if lr_dict[w] >= lr_bar and ls_dict[w] < ls_bar: # fcw
            fcw_dict[w] = normalize(lr_dict[w],lr_min,lr_max) / normalize(ls_dict[w],ls_min,ls_max)
        if lr_dict[w] < lr_bar and ls_dict[w] < ls_bar: # iw
            iw_dict[w] = 1 / (normalize(lr_dict[w],lr_min,lr_max) * normalize(ls_dict[w],ls_min,ls_max))
    ccw_sorted = [p[0] for p in sorted(ccw_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)]
    scw_sorted = [p[0] for p in sorted(scw_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)]
    fcw_sorted = [p[0] for p in sorted(fcw_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)]
    iw_sorted = [p[0] for p in sorted(iw_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)]
    kws_dict = {'lr': words_lr_sorted,
                'ls': words_ls_sorted,
                'ccw': ccw_sorted,
                'scw': scw_sorted,
                'fcw': fcw_sorted,
                'iw':iw_sorted}
    return kws_dict


def global_role_kws_extraction(global_lr_dict, global_ls_dict, labels):
    """
    version3：尝试的第三种方案，通过分位数这种大家好接受的标准来划分集合，然后归一化，然后排序。目前来看效果不错。
    根据得分的分位数进行划分高低的标准，可选择Q1，Q2，Q3三种分位数。
    通用类别指示词（CCW）：高lr跟高ls的交集
    特殊类别指示词（SCW）：低lr跟高ls的交集
    噪音类别指示词（FCW）：高lr跟低ls的交集
    类别无关词（IW）：低lr跟低ls的交集
    划分四种类别之后，每个集合先进行数值归一化，然后进行乘积或者除法的排序。
    或者先简单点儿，每个集合内部，按看lr高的，就按照lr排序，lr低的就按照-lr排序
    """
    puncs = ",./;\`~<>?:\"，。/；‘’“”、｜《》？～· \n[]{}【】「」（）()0123456789０１２３４５６７８９" \
            "，．''／；\｀～＜＞？：＂,。／;‘’“”、|《》?~·　\ｎ［］｛｝【】「」("")（） "
    kws_dict = {}
    for label in labels:
        lr_dict, ls_dict = global_lr_dict[label], global_ls_dict[label]
        lr_bar = get_quartiles(list(lr_dict.values()))
        ls_bar = get_quartiles(list(ls_dict.values()))
        lr_min, lr_max = min(list(lr_dict.values())), max(list(lr_dict.values()))
        ls_min, ls_max = min(list(ls_dict.values())), max(list(ls_dict.values()))

        words_lr_sorted = [p[0] for p in sorted(lr_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)]
        words_ls_sorted = [p[0] for p in sorted(ls_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)]

        ccw_dict, scw_dict, fcw_dict, iw_dict = {}, {}, {}, {}
        for w in ls_dict:  # todo：是乘除法好，还是加减法好？
            if lr_dict[w] >= lr_bar['Q1'] and ls_dict[w] >= ls_bar['Q1']:  # ccw
                ccw_dict[w] = normalize(lr_dict[w], lr_min, lr_max) * normalize(ls_dict[w], ls_min, ls_max)
            if lr_dict[w] < lr_bar['Q3'] and ls_dict[w] >= ls_bar['Q1']:  # scw
                scw_dict[w] = normalize(ls_dict[w], ls_min, ls_max) - normalize(lr_dict[w], lr_min, lr_max)
            if lr_dict[w] >= lr_bar['Q1'] and ls_dict[w] < ls_bar['Q3']:  # fcw
                fcw_dict[w] = normalize(lr_dict[w], lr_min, lr_max) - normalize(ls_dict[w], ls_min, ls_max)
            if lr_dict[w] < lr_bar['Q3'] and ls_dict[w] < ls_bar['Q3']:  # iw
                iw_dict[w] = 1 / (normalize(lr_dict[w], lr_min, lr_max) * normalize(ls_dict[w], ls_min, ls_max))
        ccw_sorted = [p[0] for p in sorted(ccw_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)]
        scw_sorted = [p[0] for p in sorted(scw_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)]
        fcw_sorted = [p[0] for p in sorted(fcw_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)]
        iw_sorted = [p[0] for p in sorted(iw_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)]
        kws_dict[label] = {'lr': words_lr_sorted,
                            'ls': words_ls_sorted,
                            'ccw': ccw_sorted,
                            'scw': scw_sorted,
                            'fcw': fcw_sorted,
                            'iw': iw_sorted}
    return kws_dict


