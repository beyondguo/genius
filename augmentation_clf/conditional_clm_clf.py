"""
Conditional-CLM Aug for Classification

example script:
python conditional_clm_clf.py --dataset_name sst2new_50 --clm_model_path ../saved_models/c-clm/sst2new_50_gpt2

"""

from transformers import pipeline
import pandas as pd
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--dataset_name', type=str, default='bbc_50', help='dataset dir name')
parser.add_argument('--clm_model_path', type=str, default='distilgpt2', help='gpt2, distilgpt2')
parser.add_argument('--n_aug', type=int, default=1, help='how many times to augment')
parser.add_argument('--output_name', type=str, default=None, help='output filename')
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


print(f'c-clm model: {args.clm_model_path}')
generator = pipeline("text-generation", model=args.clm_model_path, framework='pt', device=args.device)
tokenizer = generator.tokenizer

new_contents = []
new_labels = []
num_for_each_label = (len(labels) * args.n_aug) // len(label2desc)

# batch_size=50
# for label in set(labels):
#     prompt = label2desc[label]+': '
#     for i in tqdm(range(num_for_each_label//batch_size)):
#         res = generator(prompt, num_beams=3, do_sample=True, num_return_sequences=batch_size, 
#         max_length=200, pad_token_id=tokenizer.eos_token_id)
#         generated_contents = [each['generated_text'] for each in res]
#         new_contents += generated_contents
#         new_labels += [label] * batch_size


from torch.utils.data import Dataset
class List2Dataset(Dataset):
    def __init__(self, inputs):
        # inputs: list of strings
        # this class is for huggingface pipeline batch inference
        self.inputs = inputs
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, i):
        return self.inputs[i]

for label in set(labels):
    prompt = label2desc[label]+': '
    prompt_list = [prompt] * num_for_each_label
    input_dataset = List2Dataset(prompt_list)
    for out in tqdm(generator(
        input_dataset, num_beams=3, do_sample=True, 
        max_length=200, batch_size=64, 
        pad_token_id=tokenizer.eos_token_id)):
        generated_text = out[0]['generated_text']
        new_contents.append(generated_text)
        new_labels.append(label)


# remove the label prefix:
cleaned_new_contents = []
for content, label in zip(new_contents, new_labels):
    cleaned_new_contents.append(content.replace(label2desc[label]+': ', ''))

new_contents = cleaned_new_contents
augmented_contents = contents + new_contents
augmented_labels = labels + new_labels
assert len(augmented_contents) == len(augmented_labels), 'wrong num'
if args.output_name is None:
    args.output_name = f"cclm_{args.clm_model_path.split('/')[-1]}_aug{args.n_aug}"
augmented_dataset = pd.DataFrame({'content':augmented_contents, 'label':augmented_labels})
augmented_dataset.to_csv(f'../data_clf/{args.dataset_name}/{args.output_name}.csv')


print(f'>>> saved to ../data_clf/{args.dataset_name}/{args.output_name}.csv')
print(f'>>> before augmentation: {len(contents)} samples.')
print(f'>>> after augmentation: {len(augmented_contents)} samples.')
