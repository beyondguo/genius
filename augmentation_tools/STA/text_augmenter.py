import os
import jieba
import nltk
import random
from random import shuffle
import pickle
from logging import Logger
import nltk
nltk.download('wordnet')
nltk.download('punkt')
from nltk.corpus import wordnet
logger = Logger('text augmenter')
random.seed(1)

def small_fix(text):
    # 处理将词直接拼接造成的一些小问题
    puncs = ',.，。!?！？;；、'
    for punc in puncs: 
        text = text.replace(' '+punc, punc)
    return text

class TextAugmenter:
    def __init__(self, lang, using_wordnet=False):
        assert lang in ['zh', 'en'], "only support 'zh'(for Chinese) or 'en'(for English)"
        language = 'English' if lang == 'en' else 'Chinese'
        print(f'Language: {language}')
        self.lang = lang
        self.stop_words = []
        self.using_wordnet = using_wordnet
        self.joint_str = ' '
        if self.using_wordnet:
            print('Oh Wordnet!')
        # 中文
        if lang == 'zh':
            if not self.using_wordnet:
                with open(os.path.join(os.path.dirname(__file__),'weights','zh_similars_dict.pkl'),'rb') as f:
                    self.similar_words_dict = pickle.load(f)
            f = open(os.path.join(os.path.dirname(__file__),'stopwords','zh_stopwords.txt'), encoding='utf8')
            for stop_word in f.readlines():
                self.stop_words.append(stop_word[:-1])
            self.joint_str = ''
            
        # 英文
        else:  
            if not self.using_wordnet:
                with open(os.path.join(os.path.dirname(__file__),'weights','en_similars_dict.pkl'),'rb') as f:
                    self.similar_words_dict = pickle.load(f)
            
            f = open(os.path.join(os.path.dirname(__file__),'stopwords','en_stopwords.txt'), encoding='utf8')
            for stop_word in f.readlines():
                self.stop_words.append(stop_word[:-1])

        if not self.using_wordnet:
            self.vocab = list(self.similar_words_dict.keys())
    
    def tokenizer(self, text):
        if self.lang == 'zh':
            return jieba.lcut(text)
        if self.lang == 'en':
            return nltk.tokenize.word_tokenize(text)
    
    def get_similar_words(self, word):
        """
        使用预选保存好的相似词典直接查询
        若词典中没有该词，则分情况讨论：
        1. 对于中文，则查询该词的最后一个字，因为考虑到中文词中多数情况后面的字更能代表该词
        2. 对于英文，则直接返回[]
        """
        if self.using_wordnet:  # 使用wordnet查近义词
            assert self.lang == 'en', "wordnet only support 'en'"
            word = word.lower()
            similars = set()
            for w in wordnet.synsets(word):
                for l in w.lemmas():
                    sim = l.name().replace("_", " ").replace("-", " ").lower()
                    sim = "".join([char for char in sim if char in ' qwertyuiopasdfghjklzxcvbnm'])
                    similars.add(sim)
            if word in similars:
                similars.remove(word)
            return list(similars)
        
        if word not in self.vocab:
            if self.lang == 'zh':
                return self.similar_words_dict.get(word[-1], [])  # todo: 这么做是否好还有待验证
            else:
                return []
        else:
            return self.similar_words_dict.get(word)
        


    ########################################################################
    # 同义词替换
    # 替换一个语句中的n个单词为其同义词
    # random模式：随机挑选n个词进行替换
    # selective模式：指定某些词进行替换
    ########################################################################
    def aug_by_replacement(self, text, p, mode='random', selected_words=[], print_info=False):
        words = self.tokenizer(text)
        assert mode in ['random', 'selective'], "mode must be 'random' or 'selective'"
        n = max(1, int(p * len(words)))
        new_words = words.copy()
        if mode == 'selective' and len(selected_words) > 0:
            # selected_words不一定都是本文中的，可能是给定的一个大集合，所以我们要先筛选出在本文中的词
            selected_words = list(set(selected_words).intersection(set(words)))

            random.shuffle(selected_words)
            replacement_word_list = selected_words[:]  # 这里一定要使用[:]来取值，不然会改变原始词表，影响后续使用！
            if len(selected_words) < n:  # 如果数量不够，则补上一些随机词
                try:
                    replacement_word_list += random.sample(
                        list([word for word in words if word not in self.stop_words+selected_words]),
                        n - len(selected_words))
                except:
                    pass # todo: 加一个处理
        else:
            replacement_word_list = list(set([word for word in words if word not in self.stop_words]))
        random.shuffle(replacement_word_list)
        num_replaced = 0
        replacement_res = []  # 记录替换的情况
        for word in replacement_word_list:
            similars = self.get_similar_words(word)
            if len(similars) >= 1:
                similar = random.choice(similars)
                new_words = [similar if w == word else w for w in new_words]
                replacement_res.append((word, similar))
                num_replaced += 1

            if num_replaced >= n:
                break
        if print_info:
            print('replacement info:', replacement_res)
        return small_fix(self.joint_str.join(new_words))

    ########################################################################
    # 插入
    # 在语句中插入n个词
    # random模式：随机选择n个词，插入其同义词
    # selective模式：指定n个词，插入其同义词
    # given模式：直接插入给定的n个词
    ########################################################################
    def aug_by_insertion(self, text, p, mode='random', selected_words=[], print_info=False):
        words = self.tokenizer(text)
        if len(words) == 0:
            return text
        n = max(1, int(p * len(words)))
        assert mode in ['random', 'selective', 'given'], "mode must be 'random', 'selective' or 'given'"
        new_words = words.copy()
        insertion_res = []  # 记录插入的过程
        if mode == 'random':
            for i in range(n):
                word_to_insert = self.add_word(new_words, mode)
                insertion_res.append(word_to_insert)
        else:
            if mode == 'selective':
                # selected_words不一定都是本文中的，可能是给定的一个大集合，所以我们要先筛选出在本文中的词
                selected_words = list(set(selected_words).intersection(set(words)))
            random.shuffle(selected_words)
            if n > len(selected_words):  # 当given_words的数量不够时，用random来凑
                for given_word in selected_words:
                    word_to_insert = self.add_word(new_words, mode, given_word)
                    insertion_res.append(word_to_insert)
                for i in range(n - len(selected_words)):
                    word_to_insert = self.add_word(new_words, 'random')
                    insertion_res.append(word_to_insert)
            else:  # 否则，只插入n个
                for i in range(n):
                    word_to_insert = self.add_word(new_words, mode, selected_words[i])
                    insertion_res.append(word_to_insert)
        if print_info:
            print('insertion info:', insertion_res)
        return small_fix(self.joint_str.join(new_words))

    def add_word(self, words, mode, given_word=None):
        random_idx = random.randint(0, len(words) - 1)
        if mode == 'given' and given_word is not None:  # 此时插入的就是这个given word
            word_to_insert = given_word
            insert_pair = ('', given_word)
        elif mode == 'selective' and given_word is not None:  # 此时插入的是这个given word的近义词
            similars = self.get_similar_words(given_word)
            if len(similars) == 0:  # 如果当前这个词没有近义词，直接跳过
                return ('', '')
            word_to_insert = random.choice(similars)
            insert_pair = (given_word, word_to_insert)
        else:
            similars = []
            counter = 0
            while len(similars) < 1:
                random_word = words[random.randint(0, len(words) - 1)]
                similars = self.get_similar_words(random_word)
                counter += 1
                if counter >= 10:
                    return ('', '')
            word_to_insert = random.choice(similars)
            insert_pair = (random_word, word_to_insert)
        words.insert(random_idx, word_to_insert)
        return insert_pair

    ########################################################################
    # 位置互换
    # random模式：随机互换n对词的位置
    # selective模式：指定n个词，与其他词互换位置
    ########################################################################
    def aug_by_swap(self, text, p, mode='random', selected_words=[], print_info=False):
        words = self.tokenizer(text)
        if len(words) == 0:
            return text
        n = max(1, int(p * len(words)))
        assert mode in ['random', 'selective'], "mode must be 'random' or 'selective'"
        new_words = words.copy()
        swap_res = []
        if mode == 'random':
            for _ in range(n):
                new_words, swap_word = self.swap_word(new_words, mode)
                swap_res.append(swap_word)
        else:
            # selected_words不一定都是本文中的，可能是给定的一个大集合，所以我们要先筛选出在本文中的词
            selected_words = list(set(selected_words).intersection(set(words)))

            random.shuffle(selected_words)
            if n > len(selected_words):  # 数量不够，random来凑
                for selected_word in selected_words:
                    new_words, swap_word = self.swap_word(new_words, mode, selected_word)
                    swap_res.append(swap_word)
                for i in range(n - len(selected_words)):
                    new_words, swap_word = self.swap_word(new_words, 'random')
                    swap_res.append(swap_word)
            else:
                for i in range(n):
                    new_words, swap_word = self.swap_word(new_words, mode, selected_words[i])
                    swap_res.append(swap_word)
        if print_info:
            print('swap info:', swap_res)
        return small_fix(self.joint_str.join(new_words))

    def swap_word(self, words, mode, selected_word=None):
        if mode == 'selective' and selected_word is not None and selected_word in words:
            idx_1 = words.index(selected_word)  # 待操作的词的位置
        else:
            idx_1 = random.randint(0, len(words) - 1)
        idx_2 = idx_1
        counter = 0
        while idx_2 == idx_1:
            idx_2 = random.randint(0, len(words) - 1)
            counter += 1
            if counter > 3:
                return words, ''
        words[idx_1], words[idx_2] = words[idx_2], words[idx_1]
        return words, words[idx_2]  # 把原来被交换的词给返回

    ########################################################################
    # 删除
    # 以概率p删除语句中的词
    # random模式：随机删除占比p的词
    # selective模式：删除给定的那些词
    ########################################################################
    def aug_by_deletion(self, text, p, mode='random', selected_words=[], print_info=False):
        """
        p：每个词以p的概率被删除
        """
        words = self.tokenizer(text)
        if len(words) == 0:
            return text
        assert mode in ['random', 'selective'], "mode must be 'random' or 'selective'"
        words_been_deleted = []
        if len(words) == 1:
            return words
        if mode == 'random':
            new_words = []
            for word in words:
                r = random.uniform(0, 1)
                if r > p:
                    new_words.append(word)
                else:
                    words_been_deleted.append(word)
        else:  # 针对性删除难以控制数量，所以这里就控制一个上限吧
            # selected_words不一定都是本文中的，可能是给定的一个大集合，所以我们要先筛选出在本文中的词
            selected_words = list(set(selected_words).intersection(set(words)))

            random.shuffle(selected_words)
            n = int(p * len(words))
            new_words = []
            for word in words:
                # todo: 一个很明显的问题，这样去删除，出现在文本后面的词就删不到了。因为前面会有很多重复词。
                #  所以前面会被删除一大堆标点符号，二后面不受影响。
                #  感觉更好的办法是，在selected words中的，就按照p的概率删除。无非就是这种方法没法完全跟random词数对齐
                #  但不一定非要对齐，我们的目的是最终的效果好
                if word in selected_words and len(words_been_deleted) < n and word not in words_been_deleted:  # 最多删n个词, 控制每个词最多被删一次
                    words_been_deleted.append(word)
                    continue
                else:
                    new_words.append(word)
        if len(new_words) == 0:  # 被删没了，就随便拿一个词出来返回
            rand_int = random.randint(0, len(words) - 1)
            return [words[rand_int]]

        if print_info:
            print('deletion info:', words_been_deleted)
        return small_fix(self.joint_str.join(new_words))

    # 这个selection实际上也可以归类到deletion中
    # 具体做法就是只把样本中指定的一些词给提取出来
    # 在STA论文中，就是只把class-indicating words给挑出来
    def aug_by_selection(self, text, selected_words=[], print_info=False):
        words = self.tokenizer(text)
        if len(selected_words) == 0:  # 有时候selected words可能为空，就返回原句子
            print('No selected words provided for:', words)
            return words
        new_words = []
        for w in words:
            if w in selected_words:
                if print_info:
                    print('selection info:', w)
                new_words.append(w)
        # if print_info:
        #     print('selection info:', selected_words)
        return small_fix(self.joint_str.join(new_words))

    ########################################################################
    ########################################################################
    # 汇总：
    def random_text_augmentation(self, text, prob_dict=None, num_aug_dict=None,
                                 include_orig_sent=True, max_words=600, print_info=False):
        """
        - text, 直接给一段文本（不用分词），
        - prob_dict和num_aug_dict的示例如下，给出「修改比例」和「扩增数量」的配置参数：
          prob_dict = {'r': 0.1, 'i': 0.1, 's': 0.1, 'd': 0.1}
          num_aug_dict = {'r': 1, 'i': 1, 's': 1, 'd': 1}
        - include_orig_sent：是否要把原始文本也加进来，默认是加入的
        - print_info: 是否打印出文本增强的具体信息，默认不打印
        """
        if self.lang == 'zh':
            joint_str = ''
        else:
            joint_str = ' '
        if prob_dict is None:
            prob_dict = {'r': 0.1, 'i': 0.1, 's': 0.1, 'd': 0.1}
        if num_aug_dict is None:
            num_aug_dict = {'r': 1, 'i': 1, 's': 1, 'd': 1}

        augmented_texts = []
        method_list = []
        # n_r = max(1, int(prob_dict['r'] * num_words))
        # n_i = max(1, int(prob_dict['i'] * num_words))
        # n_s = max(1, int(prob_dict['s'] * num_words))

        # replacement:
        for _ in range(num_aug_dict['r']):
            a_words = self.aug_by_replacement(text, prob_dict['r'], mode='random', print_info=print_info)
            augmented_texts.append(joint_str.join(a_words))
            method_list.append('r')

        # insertion:
        for _ in range(num_aug_dict['i']):
            a_words = self.aug_by_insertion(text, prob_dict['i'], mode='random', print_info=print_info)
            augmented_texts.append(joint_str.join(a_words))
            method_list.append('i')

        # swap:
        for _ in range(num_aug_dict['s']):
            a_words = self.aug_by_swap(text, prob_dict['s'], mode='random', print_info=print_info)
            augmented_texts.append(joint_str.join(a_words))
            method_list.append('s')

        # deletion:
        for _ in range(num_aug_dict['d']):
            a_words = self.aug_by_deletion(text, prob_dict['d'], mode='random', print_info=print_info)
            augmented_texts.append(joint_str.join(a_words))
            method_list.append('d')

        # you can choose whether add original text:
        if include_orig_sent:
            augmented_texts.append(text)
            method_list.append('no')

        assert len(augmented_texts) == len(method_list), 'not same length!'
        z = list(zip(augmented_texts, method_list))
        shuffle(z)
        augmented_texts, method_list = zip(*z)
        return list(augmented_texts), list(method_list)

    def selective_text_augmentation(self, text, role_kws_dict, prob_dict=None, num_aug_dict=None,
                                    include_orig_sent=True, print_info=False):
        """
        - text, 直接给一段文本（不用分词），
        - role_kws_dict: {'CW':list ,'FW_in':list, 'FW_out':list, 'IW':list}
          CW: class-indicating words, used for replacement, inner insertion, swap, positive selection
          FW_in: fake class-indicating words(FWs), used for noise deletion
          FW_out: FWs from other classes, used for outer insertion.
        - prob_dict和num_aug_dict的示例如下，给出「修改比例」和「扩增数量」的配置参数：
          prob_dict = {'r': 0.1, 'i': 0.1, 's': 0.1, 'd': 0.2}
          {'r': 1, 'ii': 1, 'oi': 1, 's': 1, 'd': 1, 'sl': 1}
        - include_orig_sent：是否要把原始文本也加进来，默认是加入的
        - print_info: 是否打印出文本增强的具体信息，默认不打印
        """
        if self.lang == 'zh':
            joint_str = ''
        else:
            joint_str = ' '
        if prob_dict is None:
            prob_dict = {'r': 0.1, 'i': 0.1, 's': 0.1, 'd': 0.2}  # 这里的deletion可以大一些，因为要删除无关信息
        if num_aug_dict is None:
            num_aug_dict = {'r': 1, 'ii': 1, 'oi': 1, 's': 1, 'd': 1, 'sl': 1}
        words = [w for w in jieba.lcut(text) if w != ' ']
        num_words = len(words)

        augmented_texts = []
        method_list = []
        n_r = max(1, int(prob_dict['r'] * num_words))
        n_i = max(1, int(prob_dict['i'] * num_words))
        n_s = max(1, int(prob_dict['s'] * num_words))

        # replace the 'class words' by their similar words
        for _ in range(num_aug_dict['r']):
            a_words = self.aug_by_replacement(words, n_r, mode='selective', selected_words=role_kws_dict['CW'], print_info=print_info)
            augmented_texts.append(joint_str.join(a_words))
            method_list.append('r')

        # insert from inside: insert the similar words of 'class words'
        for _ in range(num_aug_dict['ii']):
            a_words = self.aug_by_insertion(words, n_i, mode='selective', given_words=role_kws_dict['CW'], print_info=print_info)
            augmented_texts.append(joint_str.join(a_words))
            method_list.append('ii')
        # insert from outside: insert out_noise_words:
        for _ in range(num_aug_dict['oi']):
            a_words = self.aug_by_insertion(words, n_i, mode='given', given_words=role_kws_dict['FW_out'], print_info=print_info)
            augmented_texts.append(joint_str.join(a_words))
            method_list.append('oi')

        # selective swap:
        for _ in range(num_aug_dict['s']):
            a_words = self.aug_by_swap(words, n_s, mode='selective', selected_words=role_kws_dict['CW'], print_info=print_info)
            augmented_texts.append(joint_str.join(a_words))
            method_list.append('s')

        # delete the inner_noise_words
        for _ in range(num_aug_dict['d']):
            a_words = self.aug_by_deletion(words, prob_dict['d'], mode='selective', selected_words=role_kws_dict['FW_in'], print_info=print_info)
            augmented_texts.append(joint_str.join(a_words))
            method_list.append('d')

        # select out the class words to form a positive sample:
        for _ in range(num_aug_dict['sl']):
            a_words = self.aug_by_selection(words,  selected_words=role_kws_dict['CW'], print_info=print_info)
            augmented_texts.append(joint_str.join(a_words))
            method_list.append('sl')

        # you can choose whether add original text:
        if include_orig_sent:
            augmented_texts.append(text)
            method_list.append('no')

        assert len(augmented_texts) == len(method_list), 'not same length!'
        z = list(zip(augmented_texts, method_list))
        shuffle(z)
        augmented_texts, method_list = zip(*z)
        return list(augmented_texts), list(method_list)


if __name__ == "__main__":  # for test
    ta = TextAugmenter('zh')
    # sentence = 'A B C D E F'
    # words = ['A', 'B', 'C', 'D', 'E', 'F']
    # selected_words = ['A', 'B', 'C']
    # print(ta.random_text_augmentation(sentence, print_info=True))
    # print('-----------')
    # role_kws_dict = {'CW': ['A', 'B'], 'FW_in': ['C', 'D'], 'FW_out': ['E', 'F']}
    # print(ta.selective_text_augmentation(sentence, role_kws_dict, print_info=True))

    s = '日前，香港商业巨人李嘉诚接受了有关媒体的专访。在全球24个国家都有投资的李嘉诚，对祖国大陆、香港、台湾的经济发展有他独到的\
    看法。他表示，台湾和香港两地的经济发展，长远将很难与祖国大陆竞争。'
    prob_dict = {'r': 0.3, 'i': 0.3, 's': 0.3, 'd': 0.3}
    num_aug_dict = {'r': 1, 'i': 1, 's': 1, 'd': 1}
    print(ta.random_text_augmentation(s, prob_dict, num_aug_dict, print_info=True))