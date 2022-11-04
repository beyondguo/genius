# Selective Text Augmentation with Word Roles

Code and demo for our paper [Selective Text Augmentation with Word Roles for Low-Resource Text Classification](https://arxiv.org/abs/2209.01560)。

![sta-process](images/sta-process.png)

Abstract:
Data augmentation techniques are widely used in text classification tasks to improve the performance of classifiers, especially in low-resource scenarios. Most previous methods conduct text augmentation without considering the different functionalities of the words in the text, which may generate unsatisfactory samples. Different words may play different roles in text classification, which inspires us to strategically select the proper roles for text augmentation. In this work, we first identify the relationships between the words in a text and the text category from the perspectives of statistical correlation and semantic similarity and then utilize them to divide the words into four roles -- Gold, Venture, Bonus, and Trivial words, which have different functionalities for text classification. Based on these word roles, we present a new augmentation technique called STA (Selective Text Augmentation) where different text-editing operations are selectively applied to words with specific roles. STA can generate diverse and relatively clean samples, while preserving the original core semantics, and is also quite simple to implement. Extensive experiments on 5 benchmark low-resource text classification datasets illustrate that augmented samples produced by STA successfully boost the performance of classification models which significantly outperforms previous non-selective methods, including two large language model-based techniques. Cross-dataset experiments further indicate that STA can help the classifiers generalize better to other datasets than previous methods.

---
# How to use

## Extracting the `Word Roles`
Extracting the word roles:
```python
from keywords_extractor import KeywordsExtractor
KE = KeywordsExtractor(lang='en')
# provide contents and corresponding labels, the `role keywords` can be extracted with one line of code:
kws_dict = KE.global_role_kws_extraction_one_line(contents, labels)
# look at the extracted role keywords for certain category:
category = 'tech'
for role in ['ccw','scw','fcw','iw']:
    print(f"{role}: {kws_dict['global_roles'][category][role][:10]}")
```
output:
```shell
keywords for "tech":
ccw (Gold): ['software', 'computer', 'PC', 'devices', 'gadget', 'Internet', 'broadband', 'video', 'images', 'technologies']
scw (Bonus): ['manufacturing', 'Google', 'telecoms', 'modern', 'energy', 'art', 'business', 'Hollywood', 'Chinese', 'businesses']
fcw (Venture): ['distribute', 'improve', 'managing', 'listen', 'households', 'downloaded', 'sharing', 'Currently', 'broadcaster', 'severe']
iw (Trivial): ['had', 'his', 'singer', 'Sunday', 'rights', 'third', 'side', 'actions', 'second', 'spokesman']
```


## Text-editing-based Augmentation
```python
from text_augmenter import TextAugmenter
TA = TextAugmenter(lang='en')
```

The `TextAugmenter` class provides unified APIs for common text-editing operations (deletion, replacement, insertion, swap, selection)
- `.aug_by_deletion(text, p, mode, selected_words)`
- `.aug_by_replacement(text, p, mode, selected_words)`
- `.aug_by_insertion(text, p, mode, selected_words)`
- `.aug_by_swap(text, p, mode, selected_words)`
- `.aug_by_selection(text, selected_words)`

Apart from `.aug_by_selection()`, all methods can specify `mode='random'` or `mode='selective'` to choose whether to use "random" augmentation or "selective" augmentation (our proposed method).

Let's say we want to augment the sentence, which is labeled as "business":
```python
sentence = "Bank of America has been banned from suing Parmalat, the food group which went bust in 2003 after an accounting scandal"
category = "business"
```

### 1. Traditional **random text augmentation** (such as [EDA]())
```python
p = 0.1
print(' '.join(TA.aug_by_deletion(text=sentence,p=p,mode='random')))
print(' '.join(TA.aug_by_replacement(text=sentence,p=p,mode='random')))
print(' '.join(TA.aug_by_insertion(text=sentence,p=p,mode='random')))
print(' '.join(TA.aug_by_swap(text=sentence,p=p,mode='random')))
```
output:
```shell
Bank of America has been banned from suing Parmalat , the food group which went bust in 2003 after an accounting scandal
Bank of America has been banned from suing Parmalat , the meal group which went bust in 2003 after an accounting scandal
Bank of America has been banned from suing Parmalat becom_ing , the food group which went bust in food_pantries_shelters 2003 after an accounting scandal
suing of America has been in from Bank Parmalat , the food group which went bust banned 2003 after an accounting scandal
```

### 2. **Selective text augmentation**
Simply change the `mode` to "selective", and specify some words:
```python
print(' '.join(TA.aug_by_deletion(text=sentence,p=p,mode='selective',selected_words=['food','banned'])))
print(' '.join(TA.aug_by_replacement(text=sentence,p=p,mode='selective',selected_words=['food','banned'])))
print(' '.join(TA.aug_by_insertion(text=sentence,p=p,mode='selective',selected_words=['food','banned'])))
print(' '.join(TA.aug_by_swap(text=sentence,p=p,mode='selective',selected_words=['food','banned'])))
print(' '.join(TA.aug_by_selection(text=sentence, selected_words=['Bank','accounting'])))
```
output:
```shell
Bank of America has been from suing Parmalat , the group which went bust in 2003 after an accounting scandal
Bank of America has been prohibiting from suing Parmalat , the nourishing_meals group which went bust in 2003 after an accounting scandal
Bank of America has been banned from suing prohibits Parmalat , the food group which nutritious_foods went bust in 2003 after an accounting scandal
Bank of America has been group from suing Parmalat , the in banned which went bust food 2003 after an accounting scandal
Bank accounting
```

Compared with **random text augmentation**,  **selective text augmentation** can focus on certain words for text-editing, therefore we can avoid some undesirable augmented samples, like
- 1) Important class-indicating words may be altered, resulting in some damage to the original meaning or even changing the label of the original text;
- 2) Unimportant words, noisy words or misleading words may be enhanced after augmentation, which may hurt the generalization ability.

### 3. **Selective text augmentation with word roles (Our proposed STA)**
Specifying the words for text-editing one sample after one sample is burdensome, our proposed `Word Roles` is here to help us *automatically* choose the words for augmentation.

By identifying the four different word roles (Gold, Venture, Bonus, and Trivial), we can specify certain roles for each augmentation operation, to generate diverse and relatively clean samples, while preserving the original core semantics.

Specifically, we use the following rules in our paper:
- selective replacement: candidates = {w|w ∈ W_venture ∪ W_bonus ∪ W_trivial }
- selective deletion: candidates = {w|w ∈ W_venture ∪ W_bonus ∪ W_trivial }
- selective insertion: candidates = {w|w ∈ W_gold ∪ W_bonus ∪ W_trivial }
- positive selection: candidates = {w|w ∈ W_gold ∪ W_trivial ∪ punctuation}

```python
# load saved keywords
import pickle
global_kws_dict_path = f'saved_keywords/global_kws_dict_{name}.pkl'
with open(global_kws_dict_path, 'rb') as f:
    global_kws_dict = pickle.load(f)

# STA:
category = 'business'
kws = global_kws_dict[category]
print(' '.join(TA.aug_by_deletion(sentence, p, 'selective', print_info=True,
                   selected_words=kws['scw']+kws['fcw']+kws['iw'])))  
print(' '.join(TA.aug_by_replacement(sentence, p, 'selective', print_info=True,
                   selected_words=kws['scw']+kws['fcw']+kws['iw'])))  
print(' '.join(TA.aug_by_insertion(sentence, p, 'selective', print_info=True,
                   selected_words=kws['ccw']+kws['scw']+kws['iw'])))  
punc_list = [w for w in ',.，。!?！？;；、']
print(' '.join(TA.aug_by_selection(sentence, print_info=True,
                    selected_words=kws['ccw']+punc_list)))
```
output:
```shell
deletion info: ['banned', 'went']
Bank of America has been from suing Parmalat , the food group which bust in 2003 after an accounting scandal
replacement info: [('went', 'drove'), ('banned', 'ban')]
Bank of America has been ban from suing Parmalat , the food group which drove bust in 2003 after an accounting scandal
insertion info: [('accounting', 'Irina_Parkhomenko_spokeswoman'), ('went', 'gone')]
Bank of America Irina_Parkhomenko_spokeswoman has been banned from suing Parmalat , the food group which went gone bust in 2003 after an accounting scandal
selection info: Parmalat
selection info: ,
selection info: bust
selection info: accounting
Parmalat , bust accounting
```


