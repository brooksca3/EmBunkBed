# Adapted from https://github.com/jamescalam/transformers/blob/main/course/training/03_mlm_training.ipynb
import sys
import torch
from creating_tokenizers.wordpiece_generator import generate, get_tokenizer
from creating_tokenizers.kmer_generator import generate_kmers
from secondary_tokenizer import ProteinTokenizer, ProteinKmerTokenizer
import time
sys.path.append('./desformers/src')

from torch.utils.checkpoint import checkpoint
from transformers2 import BertConfig, BertTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, PreTrainedTokenizerFast, BatchEncoding
from transformers2.models.bert import BertForMaskedLM


def chunk_list(data, num_chunks):
    chunk_size = len(data) // num_chunks
    remainder = len(data) % num_chunks
    chunks = []

    start_index = 0
    for i in range(num_chunks):
        # Every chunk after the remainder has been exhausted gets a standard chunk_size
        end_index = start_index + chunk_size + (1 if i < remainder else 0)
        chunks.append(data[start_index:end_index])
        start_index = end_index

    return chunks

# x = torch.load('train_inputs.pt')
# print(x.keys())
# print(len(x['input_ids']))
# print(x['input_ids'][0])
# sys.exit()


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

preload_path = 'cabrooks/10k-proteins-wordpiece'
char_tokenizer = PreTrainedTokenizerFast.from_pretrained(preload_path)

with open('/scratch/gpfs/cabrooks/deleteme_data/prepped_bunk_data/623K_truncated_512_train.txt', 'r') as f:
    text = f.read().lower().split('\n')
    # text = [t.replace("=", '') for t in text]
    text = [t[:MAXLENGTH] for t in text]
    if text[-1].strip() == '':
        text.pop(-1)
    ## we add 6 as the dummy non-suffix character
    text = ['6' + t for t in text]
    f.close()

num_train_examples = len(text)

def concatenate_encodings(encodings_list):
    concatenated = BatchEncoding({
        key: torch.cat([enc[key] for enc in encodings_list], dim=0)
        for key in encodings_list[0].keys()
    }, tensor_type='pt')
    return concatenated

start = time.time()
# Split the text into 10 chunks
text_chunks = chunk_list(text, 100)

# Tokenize each chunk
tokenized_chunks = [char_tokenizer(chunk, return_tensors='pt', max_length=MAXLENGTH, truncation=True, padding='max_length') for chunk in text_chunks]

# Set the labels for each chunk and then concatenate them
for chunk in tokenized_chunks:
    chunk['labels'] = chunk.input_ids.detach().clone()

# Concatenate all tokenized chunks
train_inputs = concatenate_encodings(tokenized_chunks)
end = time.time()
print(end - start)
print(len(train_inputs['input_ids']))
print(train_inputs['input_ids'][0])
torch.save(train_inputs, '/scratch/gpfs/cabrooks/deleteme_data/prepped_bunk_data/train_inputs_wp10k.pt')








