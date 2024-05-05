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

print('3mer combined')

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
MAXLENGTH = 1024 ## bit of buffer on top of 512 for random stuff 

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

k = 3
with open(f'creating_tokenizers/kmers_toks_{k}.txt', 'r') as f:
   toks = [t.strip() for t in f.readlines()]
   if toks[-1] == '':
      toks.pop(-1)
while len(toks) < 9989:
    toks.append(len(toks) * '~')
kmer_tokenizer = get_tokenizer(toks)

protein_tokenizer = ProteinKmerTokenizer(tokenizer=kmer_tokenizer, k=3)

preload_path = 'cabrooks/character-only-proteins'
char_tokenizer = PreTrainedTokenizerFast.from_pretrained(preload_path)
config = BertConfig()
config.vocab_size = char_tokenizer.vocab_size
config.char_tokenizer = char_tokenizer
config.char_hidden_size = 60
config.hidden_size = 768
config.max_position_embeddings = MAXLENGTH
config.secondary_tokenizers = [protein_tokenizer]

#########
model = BertForMaskedLM(config).to(device)
# Assuming you want to use the pre-trained weights from this model, use given preload_path
random_state = model.state_dict()
# wp1k model
state_model = torch.load("/scratch/gpfs/cabrooks/bunk_models/3mer_testing32/checkpoint-272356/my_custom_model.pth")
char_state_dict = torch.load('/scratch/gpfs/cabrooks/bunk_models/char_only_testing32/final-213994/my_custom_model.pth')
char_state_dict['bert.embeddings.combined_embeddings.secondary_embeddings.0.weight'] = state_model['bert.embeddings.combined_embeddings.char_embeddings.weight']
char_state_dict['bert.embeddings.word_embeddings.secondary_embeddings.0.weight'] = state_model['bert.embeddings.combined_embeddings.char_embeddings.weight']
#
char_state_dict['bert.embeddings.combined_embeddings.combination_layer.weight'] = random_state['bert.embeddings.combined_embeddings.combination_layer.weight']
char_state_dict['bert.embeddings.word_embeddings.combination_layer.weight'] = random_state['bert.embeddings.word_embeddings.combination_layer.weight']
# print(len(model['bert.embeddings.position_embeddings.weight']))
model.load_state_dict(char_state_dict)

model.to(device)
#########


print('here1')
train_inputs = torch.load('/scratch/gpfs/cabrooks/deleteme_data/prepped_bunk_data/train_inputs_char.pt')
print('here2')

for i in range(len(train_inputs['input_ids'])):
    input_ids = train_inputs['input_ids'][i]
    token_type_ids = train_inputs['token_type_ids'][i]
    attention_mask = train_inputs['attention_mask'][i]
    
    # Count non-padding tokens (assuming pad token id is 0, adjust if different)
    non_pad_count = torch.sum(input_ids != char_tokenizer.pad_token_id).item()
    
    # Calculate remainder when non-pad count is divided by 3
    remainder = non_pad_count % 3
    
    # Check how many tokens need to be added to make it divisible by 3
    if remainder == 1:
        # Need to add two special tokens to make the count divisible by 3
        special_tokens = torch.tensor(char_tokenizer.encode('6')).repeat(2)
        train_inputs['input_ids'][i] = torch.cat([special_tokens[:2], input_ids[:-2]])  # Add two tokens
        train_inputs['token_type_ids'][i] = torch.cat([torch.zeros(2, dtype=torch.long), token_type_ids[:-2]])
        train_inputs['attention_mask'][i] = torch.cat([torch.ones(2, dtype=torch.long), attention_mask[:-2]])
        train_inputs['labels'][i] = train_inputs['input_ids'][i]
    elif remainder == 2:
        # Need to add one special token to make the count divisible by 3
        special_token = torch.tensor(char_tokenizer.encode('6'))
        train_inputs['input_ids'][i] = torch.cat([special_token[:1], input_ids[:-1]])  # Add one token
        train_inputs['token_type_ids'][i] = torch.cat([torch.zeros(1, dtype=torch.long), token_type_ids[:-1]])
        train_inputs['attention_mask'][i] = torch.cat([torch.ones(1, dtype=torch.long), attention_mask[:-1]])
        train_inputs['labels'][i] = train_inputs['input_ids'][i]


M = len(train_inputs['input_ids'])  # Total number of training examples
pad_token_id = char_tokenizer.pad_token_id  # Padding token ID for input_ids
new_dim = 522  # New dimension size for each sequence

# Initialize new tensors with padding values where necessary
train_inputs_copy = {
    'input_ids': torch.full((M, new_dim), pad_token_id, dtype=torch.long),
    'token_type_ids': torch.zeros((M, new_dim), dtype=torch.long),
    'attention_mask': torch.zeros((M, new_dim), dtype=torch.long),
    'labels': torch.full((M, new_dim), pad_token_id, dtype=torch.long)
}

# Copy existing data into new tensors
for i in range(M):
    seq_len = train_inputs['input_ids'][i].size(0)
    train_inputs_copy['input_ids'][i, :seq_len] = train_inputs['input_ids'][i]
    train_inputs_copy['token_type_ids'][i, :seq_len] = train_inputs['token_type_ids'][i]
    train_inputs_copy['attention_mask'][i, :seq_len] = train_inputs['attention_mask'][i]
    train_inputs_copy['labels'][i, :seq_len] = train_inputs['input_ids'][i]  # Assuming labels are same as input_ids before padding

# Wrap as BatchEncoding
train_inputs_copy = BatchEncoding(train_inputs_copy, tensor_type='pt')
train_inputs = train_inputs_copy


# print(char_tokenizer.convert_ids_to_tokens(train_inputs['input_ids'][0]))
# sys.exit()
with open('/scratch/gpfs/cabrooks/deleteme_data/prepped_bunk_data/16K_truncated_512_validation.txt', 'r') as f:
    text_val = f.read().lower().split('\n')
    text_val = [t[:MAXLENGTH] for t in text_val]
    # text_val = [t.replace("=", '') for t in text_val]
    if text_val[-1].strip() == '':
        text_val.pop(-1)
    text_val = ['6' + t for t in text_val]
    f.close()

val_inputs = char_tokenizer(text_val, return_tensors='pt', max_length=520, truncation=True, padding='max_length')
val_inputs['labels'] = val_inputs.input_ids.detach().clone()


#######

for i in range(len(val_inputs['input_ids'])):
    input_ids = val_inputs['input_ids'][i]
    token_type_ids = val_inputs['token_type_ids'][i]
    attention_mask = val_inputs['attention_mask'][i]
    
    # Count non-padding tokens (assuming pad token id is 0, adjust if different)
    non_pad_count = torch.sum(input_ids != char_tokenizer.pad_token_id).item()
    
    # Calculate remainder when non-pad count is divided by 3
    remainder = non_pad_count % 3
    
    # Check how many tokens need to be added to make it divisible by 3
    if remainder == 1:
        # Need to add two special tokens to make the count divisible by 3
        special_tokens = torch.tensor(char_tokenizer.encode('6')).repeat(2)
        val_inputs['input_ids'][i] = torch.cat([special_tokens[:2], input_ids[:-2]])  # Add two tokens
        val_inputs['token_type_ids'][i] = torch.cat([torch.zeros(2, dtype=torch.long), token_type_ids[:-2]])
        val_inputs['attention_mask'][i] = torch.cat([torch.ones(2, dtype=torch.long), attention_mask[:-2]])
        val_inputs['labels'][i] = val_inputs['input_ids'][i]
    elif remainder == 2:
        # Need to add one special token to make the count divisible by 3
        special_token = torch.tensor(char_tokenizer.encode('6'))
        val_inputs['input_ids'][i] = torch.cat([special_token[:1], input_ids[:-1]])  # Add one token
        val_inputs['token_type_ids'][i] = torch.cat([torch.zeros(1, dtype=torch.long), token_type_ids[:-1]])
        val_inputs['attention_mask'][i] = torch.cat([torch.ones(1, dtype=torch.long), attention_mask[:-1]])
        val_inputs['labels'][i] = val_inputs['input_ids'][i]


M = len(val_inputs['input_ids'])  # Total number of training examples
pad_token_id = char_tokenizer.pad_token_id  # Padding token ID for input_ids
new_dim = 522  # New dimension size for each sequence

# Initialize new tensors with padding values where necessary
val_inputs_copy = {
    'input_ids': torch.full((M, new_dim), pad_token_id, dtype=torch.long),
    'token_type_ids': torch.zeros((M, new_dim), dtype=torch.long),
    'attention_mask': torch.zeros((M, new_dim), dtype=torch.long),
    'labels': torch.full((M, new_dim), pad_token_id, dtype=torch.long)
}

# Copy existing data into new tensors
for i in range(M):
    seq_len = val_inputs['input_ids'][i].size(0)
    val_inputs_copy['input_ids'][i, :seq_len] = val_inputs['input_ids'][i]
    val_inputs_copy['token_type_ids'][i, :seq_len] = val_inputs['token_type_ids'][i]
    val_inputs_copy['attention_mask'][i, :seq_len] = val_inputs['attention_mask'][i]
    val_inputs_copy['labels'][i, :seq_len] = val_inputs['input_ids'][i]  # Assuming labels are same as input_ids before padding

# Wrap as BatchEncoding
val_inputs_copy = BatchEncoding(val_inputs_copy, tensor_type='pt')
val_inputs = val_inputs_copy
#######

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

# training_args = TrainingArguments(
#     evaluation_strategy = "steps",
#     eval_steps=eval_every,
#     logging_steps=log_every,
#     save_steps=save_every,
#     output_dir=filestem + '/char_only_testing' + str(batch_size),
#     per_device_train_batch_size=batch_size,
#     num_train_epochs=epochs
#     # learning_rate=5e-05
# )

training_args = TrainingArguments(
    evaluation_strategy = "steps",
    eval_steps=200,
    logging_steps=200,
    save_steps=save_every,
    output_dir=filestem + '/char_3mer_combined_test' + str(batch_size),
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

model.save_pretrained(filestem + '/char_3mer_combined_test' + str(batch_size) + '/tester')

