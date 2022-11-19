For classification tasks:
Put the dataset dir in the 'data_clf' dir. The dataset dir should contain:
- a 'train.csv' file, containing the training samples, which we want to augment
- a 'test.csv' file. We will not augment the test data.
- both 'train.csv' and 'test.csv' should contain these two columns: "content" and "label". "content" is the text to feed into the model, "label" can be the index or just the label names, the label proveded will be converted to index automatically in the training scripts.

Given the dataset dir (such as 'data_clf/ng_50'), all augmentation scripts will read the 'train.csv' file (such as 'data_clf/ng_50/train.csv') and produce augmented train file into the given dataset dir (such as 'data_clf/ng_50/train_backtrans_xxx.csv').

EDA:
```shell
# use word2vec model for synonyms
python eda_clf.py --dataset_name bbc_50 --method mix --simdict w2v --p 0.1 --n_aug 2
# use wordnet for synonyms (much faster)
python eda_clf.py --dataset_name bbc_50 --method mix --simdict wordnet --p 0.1 --n_aug 4
# synonyms replacement (SR) only:
python eda_clf.py --dataset_name bbc_50 --method replace --simdict wordnet --p 0.1 --n_aug 4
```

STA:
```shell
# use word2vec model for synonyms
python sta_clf.py --dataset_name bbc_50 --method replace --simdict w2v --p 0.1 --n_aug 2
# use wordnet for synonyms (much faster)
python sta_clf.py --dataset_name bbc_50 --method replace --simdict wordnet --p 0.1 --n_aug 2
# positive selection (PS) only
python sta_clf.py --dataset_name bbc_50 --method select --simdict wordnet --p 0.1 --n_aug 2
```

Back-Translation:
```shell
python backtrans_clf.py --dataset_name bbc_50 --inter_langs de-zh --n_aug 2
```

MLM:
```shell
python mlm_clf.py --dataset_name bbc_50 --mlm_model_path distilbert-base-cased --p 0.1 --topk 5 --n_aug 2
```

C-MLM (Conditional MLM, e.g. C-BERT, BERT_prepend):
```shell
# first finetune on the training set
python conditional_mlm_finetune.py --dataset_name sst2new_50 --mlm_model_path roberta-base --num_train_epochs 15
# then run the augmentation
python conditional_mlm_clf.py --dataset_name sst2new_50 --mlm_model_path ../saved_models/c-mlm/sst2new_50_roberta-base
```

C-CLM (Conditional CLM, e.g. LAMBADA, which is a conditional GPT-2 model)
```shell
# first finetune on the training set
python conditional_clm_finetune.py --clm_model_path gpt2 --dataset_name sst2new_50 --num_train_epochs 15
# then run the augmentation
python conditional_clm_clf.py --dataset_name sst2new_50 --clm_model_path ../saved_models/c-clm/sst2new_50_gpt2
```

GeniusAug (Sketch-based Generative Augmentation, Ours):
```shell
python genius_clf.py \
    --dataset_name ng_50 \
    --genius_model_path beyond/genius-large \
    --template 4 \
    --genius_version genius-base-t4 \
    --n_aug 4 \
    --add_prompt
```

GeniusAug fine-tune on downstream datasets:
```shell
CUDA_VISIBLE_DEVICES=0 python genius_finetune.py \
    --dataset_name yahooA10k_200 \
    --checkpoint beyond/genius-large \
    --max_num_sent 5 \
    --num_train_epochs 10 \
    --batch_size 16
```

GeniusAug-mixup
```shell
python genius_mixup_clf.py \
    --dataset_name imdb_50 \
    --max_ngram 3 \
    --sketch_n_kws 15 \
    --extract_global_kws \
    --genius_version genius-mixup \
    --n_aug 4
```
