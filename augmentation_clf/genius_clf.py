"""
GeniusAug for Classification

example script:
python genius_clf.py \
    --dataset_name imdb_50 \
    --genius_model_path beyond/genius-large \
    --template 4 \
    --add_prompt \
    --genius_version genius-base-t4 \
    --n_aug 2

"""
import sys
sys.path.append('../')
from transformers import pipeline
import pandas as pd
from genius_utils import SketchExtractor, List2Dataset, setup_seed, get_stopwords
setup_seed(5)
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--dataset_name', type=str, default='bbc_50', help='dataset dir name')
parser.add_argument('--genius_model_path', type=str, default=None, help='genius model path')
parser.add_argument('--template', type=int, help='1,2,3,4')
parser.add_argument('--max_ngram', type=int,default=3, help='3 for normal passages. If the text is too short, can be set smaller')
parser.add_argument('--aspect_only', action='store_true',default=False, help='')
parser.add_argument('--no_aspect', action='store_true',default=False, help='')
parser.add_argument('--max_length', type=int, default=200, help='')
parser.add_argument('--add_prompt', action='store_true', default=True, help='if set, will prepend label prefix to sketches')
parser.add_argument('--n_aug', type=int, default=1, help='how many times to augment')
parser.add_argument('--genius_version', type=str, help='to custom output filename')
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
    checkpoint = ''
print('genius checkpoint:', checkpoint)
genius = pipeline('text2text-generation', model=checkpoint, device=args.device, framework='pt')


sketcher = SketchExtractor(model='bert')
"""
example:
s = contents[1]
aspect_keywords = ['sport']
sketcher.get_sketch(s, max_ngram=3, top=20, aspect_keywords=aspect_keywords, use_aspect_as_doc_embedding=False, template=4)
"""

def my_topk(text):
    l = len(text.split(' ')) # 使用空格简易分词俩统计长度
    # 一般为1/5的词，最少1个词，最多40个词
    return min(max(l//5, 1),40)

print('Extracting sketches...')
stopwords = get_stopwords()
sketches = []
for content, label in zip(tqdm(contents), labels):
    prompt = ''
    if args.add_prompt:
        prompt = label2desc[label]+': '
    aspect_keywords = None if args.no_aspect else label2desc[label].split(' ')
    topk = my_topk(content)
    kws = sketcher.get_kws(
        content, max_ngram=args.max_ngram,top=topk, 
        aspect_keywords=aspect_keywords, 
        use_aspect_as_doc_embedding=args.aspect_only, 
        )[1]
    kws = [w for w in kws if w not in stopwords]
    sketch = sketcher.get_sketch_from_kws(content, kws, template=args.template)
    sketch = prompt + sketch
    sketches.append(sketch)
sketch_dataset = List2Dataset(sketches)


print('Generating new samples...')
new_contents = []
for _ in range(args.n_aug):
    for out in tqdm(genius(
            sketch_dataset, num_beams=3, do_sample=True, 
            num_return_sequences=1, max_length=args.max_length, 
            batch_size=50, truncation=True)):
        generated_text = out[0]['generated_text']
        new_contents.append(generated_text)
new_labels = labels * args.n_aug
all_sketches = sketches * args.n_aug

# filter out the prompts:
if args.add_prompt:
    new_contents = [c.replace(label2desc[l]+': ', '') for c,l in zip(new_contents, new_labels)]


augmented_contents = contents + new_contents
augmented_labels = labels + new_labels
corresponding_sketches = ['ORIGINAL-SAMPLE'] * len(labels) + all_sketches
assert len(augmented_contents) == len(augmented_labels), 'wrong num'
assert len(augmented_contents) == len(corresponding_sketches), 'wrong num'
args.output_name = f"genius_prompt{args.add_prompt}_asonly_{args.aspect_only}_{args.genius_version}_aug{args.n_aug}"
augmented_dataset = pd.DataFrame({'content':augmented_contents, 'sketch':corresponding_sketches, 'label':augmented_labels})
augmented_dataset.to_csv(f'../data_clf/{args.dataset_name}/{args.output_name}.csv')


print(f'>>> saved to ../data_clf/{args.dataset_name}/{args.output_name}.csv')
print(f'>>> before augmentation: {len(contents)} samples.')
print(f'>>> after augmentation: {len(augmented_contents)} samples.')
