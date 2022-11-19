"""
GENIUS Fine-tuning on target dataset

To make GENIUS better suited for downstream tasks, we finetune the model with prompts.

- cut the original text into smaller chunks (less than 15 sentences)
- extract the label-aware sketch from the chunk, and prepend the corresponding label prefix to the sketch
- prepend the corresponding label prefix the each chunk
- prompt + sketch --> GENIUS --> prompt + content

example script:
CUDA_VISIBLE_DEVICES=0
python genius_finetune.py \
    --dataset_name bbc_50 \
    --checkpoint beyond/genius-large \
    --max_num_sent 15 \
    --num_train_epochs 10 \
    --batch_size 16

TODO: make sure to use the label names/descriptions
"""
import sys
sys.path.append('../')
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import Dataset as HFDataset
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from genius_utils import SketchExtractor, List2Dataset, setup_seed, get_stopwords
from rouge_score import rouge_scorer
from datasets import load_metric
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
random.seed(5)
from collections import defaultdict
import argparse
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--dataset_name', type=str, default='bbc_50', help='dataset dir name')
parser.add_argument('--checkpoint', type=str, default='', help='genius checkpoint')
parser.add_argument('--aspect_only', action='store_true', default=False, help='')
parser.add_argument('--template', type=int, default=4, help='')
parser.add_argument('--num_train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=8, help='batch size for dataloaders')
parser.add_argument('--max_num_sent', type=int, default=15, help='num of sentences for a chunk')
parser.add_argument('--comment', type=str, default='', help='to modify save name')

# parser.add_argument('--device', type=int, default=0, help='cuda device index, if not found, will switch to cpu')
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
    print(label2desc)
else:
    label2desc = {label:label for label in set(labels)}


model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint)
tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)


sketcher = SketchExtractor(model='bert')


def my_topk(text):
    l = len(text.split(' ')) # 使用空格简易分词俩统计长度
    # 一般为1/5的词，最少1个词，最多40个词
    return min(max(l//5, 1),40)


"""
目前这样的设置感觉不太合理。
那些短的文本，被你强行都拼成了15个句子。
先试试吧，如果效果不行，就改成短文本不处理，对长文本才进行切分。
"""
print('Extracting chunks and sketches...')

label2longtext = defaultdict(str)
for label, content in zip(labels, contents):
    label2longtext[label] += content

stopwords = get_stopwords()
input_texts = []
output_texts = []
max_num_sent = args.max_num_sent
for label in label2longtext:
    print('preparing for label: ', label)
    prompt = label2desc[label] + ': '
    aspect_keywords = label2desc[label].split(' ')
    longtext = label2longtext[label]
    sents = sent_tokenize(longtext)
    r = max(len(sents) // max_num_sent ,1)
    for i in tqdm(range(r)):
        span = ' '.join(sents[i * max_num_sent : (i+1) * max_num_sent])
        
        topk = my_topk(span)
        kws = sketcher.get_kws(
            content, max_ngram=3,top=topk, 
            aspect_keywords=aspect_keywords, 
            use_aspect_as_doc_embedding=args.aspect_only, 
            )[1]
        kws = [w for w in kws if w not in stopwords]
        sketch = sketcher.get_sketch_from_kws(content, kws, template=args.template)
        
        input_texts.append(prompt + sketch)
        output_texts.append(prompt + span)

text_dataset = HFDataset.from_dict({'sketch':input_texts, 'text':output_texts})

# define the inputs and labels for sketch-based reconstruction pre-training
max_input_length = 100
max_target_length = 300
print("********** Sketch type is: ", args.template)
def preprocess_function(examples):
    """
    # inputs: the sketch
    # labels: the original text
    """
    model_inputs = tokenizer(examples['sketch'], max_length=max_input_length, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['text'], max_length=max_target_length, truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_dataset = text_dataset.map(preprocess_function, batched=True)



# ROUGE metric：
rouge_score = load_metric("rouge")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Decode generated summaries into text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Decode reference summaries into text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    # Compute ROUGE scores
    result = rouge_score.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract the median scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}


##################################################################
#                     training
##################################################################


output_dir = f"../saved_models/genius_finetuned_for_{args.dataset_name}{args.comment}"

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy = 'no', # maybe set to 'no' to save time?
    save_total_limit = 1,
    fp16 = True,
    learning_rate=5.6e-5,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    weight_decay=0.01,
    num_train_epochs=args.num_train_epochs,
    predict_with_generate=True,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


trainer = Seq2SeqTrainer(
    model,
    training_args,
    train_dataset=tokenized_dataset.remove_columns(text_dataset.column_names),
    eval_dataset=tokenized_dataset.remove_columns(text_dataset.column_names), # just look at the train set
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


trainer.train(resume_from_checkpoint = False)
trainer.save_model(output_dir)

# moving model files to main dir:
# import os
# fs = os.listdir(output_dir)
# for f in fs:
#     if 'checkpoint' in f:
#         checkpoint_dir = output_dir + '/' + f
#         print(checkpoint_dir)
# if checkpoint_dir:
#     os.system(f'mv {checkpoint_dir}/* {output_dir}/')
