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

def concatenate_encodings(encodings_list):
    concatenated = BatchEncoding({
        key: torch.cat([enc[key] for enc in encodings_list], dim=0)
        for key in encodings_list[0].keys()
    }, tensor_type='pt')
    return concatenated

epochs = 75
MAXLENGTH = 520 ## bit of buffer on top of 512 for random stuff 

### ADJUST PARAMETERS AS DESIRED ###
num_logs_per_epoch = 4
num_evals_per_epoch = 4
num_saves_per_epoch = 1
mask_proportion = 0.15

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
config = BertConfig()
config.vocab_size = char_tokenizer.vocab_size
config.char_tokenizer = char_tokenizer

config.char_hidden_size = 60
config.hidden_size = 768
config.max_position_embeddings = 1024
config.secondary_tokenizers = []
model = BertForMaskedLM(config).to(device)

model.to(device)

# with open('/scratch/gpfs/cabrooks/deleteme_data/prepped_bunk_data/623K_truncated_512_train.txt', 'r') as f:
#     text = f.read().lower().split('\n')
#     # text = [t.replace("=", '') for t in text]
#     text = [t[:MAXLENGTH] for t in text]
#     if text[-1].strip() == '':
#         text.pop(-1)
#     ## we add 6 as the dummy non-suffix character
#     text = ['6' + t for t in text]
#     f.close()

# num_train_examples = len(text)
# text_chunks = chunk_list(text, 100)

# # Tokenize each chunk
# tokenized_chunks = [char_tokenizer(chunk, return_tensors='pt', max_length=MAXLENGTH, truncation=True, padding='max_length') for chunk in text_chunks]

# # Set the labels for each chunk and then concatenate them
# for chunk in tokenized_chunks:
#     chunk['labels'] = chunk.input_ids.detach().clone()
    
# train_inputs = concatenate_encodings(tokenized_chunks)

# train_inputs = char_tokenizer(text, return_tensors='pt', max_length=MAXLENGTH, truncation=True, padding='max_length')
# train_inputs['labels'] = train_inputs.input_ids.detach().clone()
print('here1')
train_inputs = torch.load('/scratch/gpfs/cabrooks/deleteme_data/prepped_bunk_data/train_inputs_char.pt')
print('here2')

with open('/scratch/gpfs/cabrooks/deleteme_data/prepped_bunk_data/16K_truncated_512_validation.txt', 'r') as f:
    text_val = f.read().lower().split('\n')
    text_val = [t[:MAXLENGTH] for t in text_val]
    # text_val = [t.replace("=", '') for t in text_val]
    if text_val[-1].strip() == '':
        text_val.pop(-1)
    text_val = ['6' + t for t in text_val]
    f.close()

val_inputs = char_tokenizer(text_val, return_tensors='pt', max_length=MAXLENGTH, truncation=True, padding='max_length')
val_inputs['labels'] = val_inputs.input_ids.detach().clone()

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: val[idx].clone().detach() for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)

train_dataset = BaseDataset(train_inputs)
val_dataset = BaseDataset(val_inputs)

data_collator = DataCollatorForLanguageModeling(tokenizer=char_tokenizer, mlm_probability=mask_proportion)
num_train_examples = len(train_inputs['input_ids'])
num_steps_per_epoch = (num_train_examples) // batch_size

log_every = int(num_steps_per_epoch/num_logs_per_epoch)
eval_every = int(num_steps_per_epoch/num_evals_per_epoch)
save_every = int(num_steps_per_epoch/num_saves_per_epoch)

training_args = TrainingArguments(
    evaluation_strategy = "steps",
    eval_steps=eval_every,
    logging_steps=log_every,
    save_steps=save_every,
    output_dir=filestem + '/char_only_testing' + str(batch_size),
    per_device_train_batch_size=batch_size,
    num_train_epochs=epochs
    # learning_rate=5e-05
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator
)

trainer.train()
model.config.char_tokenizer = None
model.config.secondary_tokenizers = None

model.save_pretrained(filestem + '/char_only_testing' + str(batch_size) + '/tester')

