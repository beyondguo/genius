# SEGA: SkEtch-based Generative Augmentation

**基于草稿的生成式增强模型**

**SEGA** is a **general text augmentation model** that can be used for data augmentation for **various NLP tasks** (including sentiment analysis, topic classification, NER, and QA). SEGA uses an encoder-decoder structure (based on the BART architecture) and is pre-trained on the `C4-realnewslike` corpus. 


![sega-illustration](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/typora/sega-main-illustration.png)

- Paper: [coming soon](to_be_added)


**SEGA** is able to write complete paragraphs given a sketch (or framework), which can be composed of:
- keywords /key-phrases, like [NLP | AI | computer science]
- spans, like [Conference on Empirical Methods | submission of research papers]
- sentences, like [I really like machine learning | I work at Google since last year]
- all mixup~

### How to use
```python
from transformers import pipeline
# 1. load the model with the huggingface `pipeline`
sega = pipeline("text2text-generation", model='beyond/sega-large', device=0)
# 2. provide a sketch (joint by <mask> tokens)
sketch = "<mask> Conference on Empirical Methods <mask> submission of research papers <mask> Deep Learning <mask>"
# 3. Do generation!
generated_text = sega(sketch, num_beams=3, do_sample=True, max_length=200)[0]['generated_text']
print(generated_text)
```
Output:
```shell
'The Conference on Empirical Methods welcomes the submission of research papers. Abstracts should be in the form of a paper or presentation. Please submit abstracts to the following email address: eemml.stanford.edu. The conference will be held at Stanford University on April 1618, 2019. The theme of the conference is Deep Learning.'
```

## Model variations


| Model | #params | Language |
|------------------------|--------------------------------|-------|
| [`sega-large`]() | xM   | English |
| [`sega-base`(coming soon)]()  | xM    | English |
| [`sega-small`(coming soon)]()        | xM    | English |
| [`sega-large-chinese`(coming soon)]() | xM    |  Chinese |
| [`sega-base-chinese`(coming soon)]() | xM    | Chinese |
| [`sega-small-chinese`(coming soon)]() | xM | Chinese |


## Data Augmentation for Text Classification Tasks:
- Setting: Low-resource setting, where only n={50,100,200,500,1000} labeled samples are available for training. The below results are the average of all training sizes.
- Datasets: [HuffPost](https://huggingface.co/datasets/khalidalt/HuffPost), [BBC](https://huggingface.co/datasets/SetFit/bbc-news), [SST2](https://huggingface.co/datasets/glue), [IMDB](https://huggingface.co/datasets/imdb), [Yahoo](https://huggingface.co/datasets/yahoo_answers_topics), [20NG](https://huggingface.co/datasets/newsgroup).
- Base classifier: [DistilBERT](https://huggingface.co/distilbert-base-cased)


### BibTeX entry and citation info
TBD

