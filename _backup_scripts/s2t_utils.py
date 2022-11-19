import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# 清洁工：
# function to remove special characters
def remove_special_characters(text):
    # 移除非字母、非数字、非主要标点
    pat = r'[^a-zA-z0-9.,!?/:;\"\'\s]' 
    text =  re.sub(pat, '', text)
    text =  re.sub('``', '', text)
    text =  re.sub(r'\s{2,}', ' ', text) # 匹配两个或更多的空白字符
    return text.strip()
def remove_brakets(text):
    text =  re.sub(r'\[(.*)\]', '', text)
    text =  re.sub(r'\((.*)\)', '', text)
    return text
def remove_last_sentence(text):
    # 当超过1个句子，且最后一个句子不以标点结尾，就移除最后一句
    sents = sent_tokenize(text)
    text = ' '.join(sents)
    if len(sents) > 1:
        if sents[-1][-1] not in ".?!。？！\'\"":
            text = ' '.join(sents[:-1])
    return text
def clean_pipeline(text):
    return remove_last_sentence((remove_special_characters(remove_brakets(text))))


stopwords = stopwords.words('english')
def get_stopwords():
    return stopwords


from torch.utils.data import Dataset
class S2T_Dataset(Dataset):
    def __init__(self, inputs):
        self.inputs = inputs
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, i):
        return self.inputs[i]