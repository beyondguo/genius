N_TRAIN = 87599
N_AUG = 1
v = 1

from datasets import load_metric
metric = load_metric("squad")

"""
format the input like this:
{'id': '56be4db0acb8001400a502ec', 'prediction_text': 'Denver Broncos'}
{'id': '56be4db0acb8001400a502ec', 'answers': {'text': ['Denver Broncos', 'Denver Broncos', 'Denver Broncos'], 'answer_start': [177, 177, 177]}}
then:
metric.compute(predictions=predicted_answers, references=theoretical_answers)
"""

# load the dataset to be filtered
import pandas as pd
aug_file_path = f'qa_data/squad_first{N_TRAIN}_aug{N_AUG}_v{v}.pkl'
aug_df = pd.read_pickle(aug_file_path)
print(f'>>>Original aug size: {len(aug_df)}')
augmented_dataset = {"context":list(aug_df['context']),"question":list(aug_df['question']),"answers":list(aug_df['answers'])}

# load the filter model
from transformers import pipeline
filter_model_path = f'saved_models/squad1_{N_TRAIN}examples_baseline'
# filter_model_path = f'saved_models/squad1_full_baseline'
filter_model = pipeline("question-answering", model=filter_model_path)

"""
filter_model(question=question, context=context)
>>>
{'score': 0.9978998899459839,
 'start': 78,
 'end': 105,
 'answer': 'Jax, PyTorch and TensorFlow'}
"""

from tqdm import tqdm
from collections import defaultdict
filtered_augmented_dataset = defaultdict(list)

for context, question, aa in zip(tqdm(augmented_dataset['context']),augmented_dataset['question'],augmented_dataset['answers']):
    prediction = filter_model(question,context)
    predicted_answer = {'id':1,'prediction_text': prediction['answer']}
    augmented_answer = {'id':1,'answers':aa}
    res = metric.compute(predictions=[predicted_answer], references=[augmented_answer])
    exact_match, f1 = res['exact_match'], res['f1']
    
    
    if exact_match == 0 or f1 == 0:
        print(res)
#         print(f'>>> C:{context}')
#         print(f'>>> Q:{question}')
#         print(f'>>> A:{aa}')
#         print(f'>>> Pred: {prediction["answer"]}')
        continue
    else:
        filtered_augmented_dataset['context'].append(context)
        filtered_augmented_dataset['question'].append(question)
        filtered_augmented_dataset['answers'].append(aa)
    
print('>>>After filtering: ',len(filtered_augmented_dataset['context']))


# save
df_filter = pd.DataFrame(filtered_augmented_dataset)
print(len(df_filter))
df_filter.to_pickle(f'qa_data/squad_first{N_TRAIN}_aug{N_AUG}_v{v}_filtered.pkl')
print(f'saved to [qa_data/squad_first{N_TRAIN}_aug{N_AUG}_v{v}_filtered.pkl]')