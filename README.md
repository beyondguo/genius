# SEGA: SkEtch-based Generative Augmentation

**基于草稿的生成式增强模型**

**SEGA** is a **general text augmentation model** that can be used for data augmentation for **various NLP tasks** (including sentiment analysis, topic classification, NER, and QA). SEGA uses an encoder-decoder structure (based on the BART architecture) and is pre-trained on the `C4-realnewslike` corpus. 


![sega-illustration](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/typora/sega-main-illustration.png)

- Paper: [coming soon](to_be_added)

- Models hosted in 🤗 Huggingface:

| Model | #params | Language |
|------------------------|--------------------------------|-------|
| [`sega-large`(paper version)](https://huggingface.co/beyond/sega-large) | xM   | English |
| [`sega-base`(coming soon)]()  | xM    | English |
| [`sega-large-chinese`(coming soon)]() | xM    |  Chinese |
| [`sega-base-chinese`(New!)](https://huggingface.co/beyond/sega-base-chinese) | xM    | Chinese |

<img src="https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/typora/sega-hf-api.jpg" width="50%" />

**SEGA** is able to write complete paragraphs given a sketch (or framework), which can be composed of:
- keywords /key-phrases, like "––NLP––AI––computer––science––"
- spans, like "Conference on Empirical Methods––submission of research papers––"
- sentences, like "I really like machine learning––I work at Google since last year––"
- or mixup~

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




## Data Augmentation for Text Classification Tasks:
- Setting: Low-resource setting, where only n={50,100,200,500,1000} labeled samples are available for training. The below results are the average of all training sizes.
- Datasets: [HuffPost](https://huggingface.co/datasets/khalidalt/HuffPost), [BBC](https://huggingface.co/datasets/SetFit/bbc-news), [SST2](https://huggingface.co/datasets/glue), [IMDB](https://huggingface.co/datasets/imdb), [Yahoo](https://huggingface.co/datasets/yahoo_answers_topics), [20NG](https://huggingface.co/datasets/newsgroup).
- Base classifier: [DistilBERT](https://huggingface.co/distilbert-base-cased)


In-distribution (ID) evaluations:
|   Method   |    Huff    |     BBC    |    Yahoo   |    20NG    |    IMDB    |    SST2    |    avg.    |
|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
|    none    |   79.17   | **96.16** |   45.77   |   46.67   |   77.87   |   76.67   |   70.39   |
|     EDA    |   79.20   |   95.11   |   45.10   |   46.15   |   77.88   |   75.52   |   69.83   |
|    BackT   |   80.48   |   95.28   |   46.10   |   46.61   |   78.35   |   76.96   |   70.63   |
|     MLM    |   80.04   |   96.07   |   45.35   |   46.53   |   75.73   |   76.61   |   70.06   |
|    C-MLM   |   80.60   |   96.13   |   45.40   |   46.36   |   77.31   |   76.91   |   70.45   |
|   LAMBADA  |   81.46   |   93.74   |   50.49   |   47.72   |   78.22   |   78.31   |   71.66   |
|     STA    |   80.74   |   95.64   |   46.96   |   47.27   |   77.88   |   77.80   |   71.05   |
|  **SEGA**  |   81.43   |   95.74   |   49.60   |   50.38   | **80.16** |   78.82   |   72.68   |
| **SEGA-f** | **81.82** |   95.99   | **50.42** | **50.81** |   79.40   | **80.57** | **73.17** |

Out-of-distribution (OOD) evaluations:
|            |  Huff->BBC |  BBC->Huff | IMDB->SST2 | SST2->IMDB |    avg.    |
|------------|:----------:|:----------:|:----------:|:----------:|:----------:|
|    none    |   62.32   |   62.00   |   74.37   |   73.11   |   67.95   |
|     EDA    |   67.48   |   58.92   |   75.83   |   69.42   |   67.91   |
|    BackT   |   67.75   |   63.10   |   75.91   |   72.19   |   69.74   |
|     MLM    |   66.80   |   65.39   |   73.66   |   73.06   |   69.73   |
|    C-MLM   |   64.94   | **67.80** |   74.98   |   71.78   |   69.87   |
|   LAMBADA  |   68.57   |   52.79   |   75.24   |   76.04   |   68.16   |
|     STA    |   69.31   |   64.82   |   74.72   |   73.62   |   70.61   |
|  **SEGA**  |   74.87   |   66.85   |   76.02   |   74.76   |   73.13   |
| **SEGA-f** | **76.18** |   66.89   | **77.45** | **80.36** | **75.22** |
### BibTeX entry and citation info
TBD

