"""
GeniusAug-mixup for Classification

example script:
python genius_mixup_clf.py \
    --dataset_name imdb_50 \
    --max_ngram 3 \
    --sketch_n_kws 15 \
    --extract_global_kws \
    --genius_version genius-mixup \
    --n_aug 4

"""
import sys
sys.path.append('../')
from transformers import pipeline
from collections import defaultdict
import pandas as pd
import random
from genius_utils import SketchExtractor, List2Dataset, setup_seed, get_stopwords
setup_seed(5)
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--dataset_name', type=str, default='yahooA10k_50', help='dataset dir name')
parser.add_argument('--genius_model_path', type=str, default=None, help='genius model path')
parser.add_argument('--template', type=int, default=4, help='1,2,3,4')
parser.add_argument('--max_ngram', type=int,default=3, help='3 for normal passages. If the text is too short, can be set smaller')
parser.add_argument('--sketch_n_kws', type=int,default=15, help='how many kewords to form the sketch')
parser.add_argument('--aspect_only', action='store_true',default=True, help='')
parser.add_argument('--extract_global_kws', action='store_true', default=True, help='')
parser.add_argument('--add_prompt', action='store_true', default=True, help='if set, will prepend label prefix to sketches')
parser.add_argument('--max_length', type=int, default=200, help='')
parser.add_argument('--n_aug', type=int, default=1, help='how many times to augment')
parser.add_argument('--genius_version', type=str, default='genius-mixup', help='to custom output filename')
parser.add_argument('--device', type=int, default=0, help='cuda device index, if not found, will switch to cpu')
args = parser.parse_args()


# read dataset
# dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data_clf', args.dataset_name, 'train.csv'))
dataset = pd.read_csv(f'../data_clf/{args.dataset_name}/train.csv')
contents = list(dataset['content'])
labels = list(dataset['label'])
# use the label names/descriptions to replace the label indices
from label_desc import get_label2desc
if get_label2desc(args.dataset_name):
    label2desc = get_label2desc(args.dataset_name)
else:
    label2desc = {label:label for label in set(labels)}
print(label2desc)

if args.genius_model_path is not None:
    checkpoint = args.genius_model_path
else:
    # checkpoint = '../saved_models/bart-base-c4-realnewslike-4templates-passage-and-sent-max15sents_2-sketch4/checkpoint-215625'
    checkpoint = '../saved_models/bart-large-c4-l_50_200-d_13799838-yake_mask-t_3900800/checkpoint-152375'
    
print('genius checkpoint:', checkpoint)
genius = pipeline('text2text-generation', model=checkpoint, device=args.device)




def my_topk(text):
    l = len(text.split(' ')) # 使用空格简易分词俩统计长度
    # 一般为1/5的词，最少1个词，最多40个词
    return min(max(l//5, 1),10)

def contain_alpha(string):
    for each in string:
        if each.isalpha():
            return True
    return False

stopwords = get_stopwords()
sketcher = SketchExtractor(model='bert')

label2kws = defaultdict(list)
if args.extract_global_kws:
    print(">>> extract_global_kws Yes!")
    label2longtext = defaultdict(str)
    for label, content in zip(labels, contents):
        label2longtext[label] += content
    for label in label2longtext:
        longtext = label2longtext[label]
        aspect_keywords = label2desc[label].split(' ')
        kws = sketcher.get_kws(
            longtext, max_ngram=args.max_ngram,top=50, 
            aspect_keywords=aspect_keywords, 
            use_aspect_as_doc_embedding=True, 
            )[1]
        kws = [w for w in kws if w not in stopwords]
        kws = [w for w in kws if contain_alpha(w)]
        label2kws[label] = kws
else:
    for content, label in zip(tqdm(contents), labels):
        aspect_keywords = label2desc[label].split(' ')
        topk = my_topk(content)
        kws = sketcher.get_kws(
            content, max_ngram=args.max_ngram,top=topk, 
            aspect_keywords=aspect_keywords, 
            use_aspect_as_doc_embedding=args.aspect_only, 
            )[1]
        kws = [w for w in kws if w not in stopwords]
        kws = [w for w in kws if contain_alpha(w)]
        label2kws[label] += kws


# augmentation:
num_for_each_label = (len(labels) * args.n_aug) // len(label2desc)
new_contents = []
new_labels = []
mixup_sketches = []
for label in label2desc:
    if args.add_prompt:
        prompt = label2desc[label]+' content: '
    else:
        prompt = ''
    for _ in range(num_for_each_label):
        candidates = random.sample(label2kws[label], min(args.sketch_n_kws,len(label2kws[label]))) 
        sketch = '<mask> ' + ' <mask> '.join(candidates) + ' <mask>'
        sketch = prompt + sketch
        mixup_sketches.append(sketch)
        new_labels.append(label)

sketch_dataset = List2Dataset(mixup_sketches)


print('Generating new samples...')
for out in tqdm(genius(
        sketch_dataset, num_beams=3, do_sample=True, 
        num_return_sequences=1, max_length=args.max_length, 
        batch_size=32, truncation=True)):
    generated_text = out[0]['generated_text']
    new_contents.append(generated_text)

# filter out the prompts:
if args.add_prompt:
    new_contents = [c.replace(label2desc[l]+' content: ', '') for c,l in zip(new_contents, new_labels)]



augmented_contents = contents + new_contents
augmented_labels = labels + new_labels
corresponding_sketches = ['ORIGINAL-SAMPLE'] * len(labels) + mixup_sketches
assert len(augmented_contents) == len(augmented_labels), 'wrong num'
assert len(augmented_contents) == len(corresponding_sketches), 'wrong num'
args.output_name = f"geniusMix_prompt{args.add_prompt}_asonly_{args.aspect_only}_{args.genius_version}_aug{args.n_aug}"
augmented_dataset = pd.DataFrame({'content':augmented_contents, 'sketch':corresponding_sketches, 'label':augmented_labels})
augmented_dataset.to_csv(f'../data_clf/{args.dataset_name}/{args.output_name}.csv')


print(f'>>> saved to ../data_clf/{args.dataset_name}/{args.output_name}.csv')
print(f'>>> before augmentation: {len(contents)} samples.')
print(f'>>> after augmentation: {len(augmented_contents)} samples.')
