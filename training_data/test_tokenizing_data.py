# Adapted from https://github.com/jamescalam/transformers/blob/main/course/training/03_mlm_training.ipynb
import sys
import torch
from creating_tokenizers.wordpiece_generator import generate, get_tokenizer
from creating_tokenizers.kmer_generator import generate_kmers
from secondary_tokenizer import ProteinTokenizer, ProteinKmerTokenizer
import time
sys.path.append('./desformers/src')

from torch.utils.checkpoint import checkpoint
from transformers2 import BertConfig, BertTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, PreTrainedTokenizerFast
from transformers2.models.bert import BertForMaskedLM

MAXLENGTH = 520 ## bit of buffer on top of 512 for random stuff 


if len(sys.argv) == 1:
        print("Received no argument for batch size. Defaulting to 32.")
        batch_size = 16
elif len(sys.argv) > 1:
        print(f"Setting batch size to {sys.argv[1]}.")
        batch_size = int(sys.argv[1])

filestem = '/scratch/gpfs/cabrooks/bunk_models/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}.")

preload_path = 'cabrooks/character-only-proteins'
char_tokenizer = PreTrainedTokenizerFast.from_pretrained(preload_path)

with open('training_data/623K_truncated_512_train.txt', 'r') as f:
    text = f.read().lower().split('\n')
    # text = [t.replace("=", '') for t in text]
    text = [t[:MAXLENGTH] for t in text]
    if text[-1].strip() == '':
        text.pop(-1)
    ## we add 6 as the dummy non-suffix character
    text = ['6' + t for t in text]
    f.close()

num_train_examples = len(text)

# train_inputs = char_tokenizer(text, return_tensors='pt', max_length=MAXLENGTH, truncation=True, padding='max_length')
# train_inputs['labels'] = train_inputs.input_ids.detach().clone()
exts = [50,100,500,1000,5000]

for ex in exts:

    start = time.time()
    train_inputs = char_tokenizer(text[:ex], return_tensors='pt', max_length=MAXLENGTH, truncation=True, padding='max_length')
    end = time.time()
    print(f"time for {ex} is {end - start}")







