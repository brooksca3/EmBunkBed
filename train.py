# Adapted from https://github.com/jamescalam/transformers/blob/main/course/training/03_mlm_training.ipynb
import sys
import torch
from creating_tokenizers.wordpiece_generator import generate, get_tokenizer
from creating_tokenizers.kmer_generator import generate_kmers
from secondary_tokenizer import ProteinTokenizer, ProteinKmerTokenizer
sys.path.append('./desformers/src')

from torch.utils.checkpoint import checkpoint
from transformers2 import BertConfig, BertTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, PreTrainedTokenizerFast
from transformers2.models.bert import BertForMaskedLM
from transformers2.data.data_collator import DataCollatorForSequenceMask

epochs = 500
MAXLENGTH = 1024

### ADJUST PARAMETERS AS DESIRED ###
num_logs_per_epoch = 1
num_evals_per_epoch = 1
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

# Fixed k-mer
k = 2 ### ADJUST PARAMETER AS DESIRED ###
protein_tokens = generate_kmers(k, 'creating_tokenizers/100_examples.txt')
protein_wp_tokenizer = get_tokenizer(protein_tokens)
pt = ProteinKmerTokenizer(k, tokenizer=protein_wp_tokenizer)
print(pt.tokenize('6VLDLADQLM'.lower()))
print(pt.tokenize('6VTT'.lower()))

sys.exit()

# WordPiece
# protein_tokens = generate(2000, 'creating_tokenizers/100_examples.txt')
# protein_wp_tokenizer = get_tokenizer(protein_tokens)
# pt = ProteinTokenizer(tokenizer=protein_wp_tokenizer)
# print(pt.tokenize('6ILDLADQL[MASK]DAADTARPAASTQSATQNTPAEPVPP'.lower()))

preload_path = 'cabrooks/character-only-proteins'
char_tokenizer = PreTrainedTokenizerFast.from_pretrained(preload_path)
config = BertConfig()
config.vocab_size = char_tokenizer.vocab_size
config.char_tokenizer = char_tokenizer
# config.wordpiece_tokenizer = wordpiece_tokenizer
config.char_hidden_size = 60
config.hidden_size = 768
config.max_position_embeddings = 1024
config.secondary_tokenizers = []
model = BertForMaskedLM(config).to(device)

# Assuming you want to use the pre-trained weights from this model, use given preload_path
# pre_trained = torch.load('my_custom_model.pth')
#random_state = model.state_dict()
#key1 = "bert.embeddings.combined_embeddings.combination_layer.weight"
#key2 = "bert.embeddings.word_embeddings.combination_layer.weight"
#for key in pre_trained:
#  if key != key1:
#    if key != key2:
#      random_state[key] = pre_trained[key]

#model.load_state_dict(random_state)
# model.load_state_dict(torch.load('my_custom_model.pth'))
model.to(device)

with open('creating_tokenizers/100_examples_prefixed.txt', 'r') as f:
  text = f.read().lower().split('\n')
  # text = [t.replace("=", '') for t in text]
  text = [t[:MAXLENGTH] for t in text]
  if text[-1].strip() == '':
      text.pop(-1)
  f.close()

num_train_examples = len(text)

train_inputs = char_tokenizer(text, return_tensors='pt', max_length=MAXLENGTH, truncation=True, padding='max_length')
train_inputs['labels'] = train_inputs.input_ids.detach().clone()

with open('creating_tokenizers/100_examples_prefixed.txt', 'r') as f:
  text_val = f.read().lower().split('\n')
  text_val = [t[:MAXLENGTH] for t in text]
  # text_val = [t.replace("=", '') for t in text_val]
  if text_val[-1].strip() == '':
      text_val.pop(-1)
  f.close()

val_inputs = char_tokenizer(text_val, return_tenfsors='pt', max_length=MAXLENGTH, truncation=True, padding='max_length')
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

num_steps = (epochs * num_train_examples) // batch_size
print(f"num steps = {num_steps}")
log_every = int(num_steps/num_logs_per_epoch)
eval_every = int(num_steps/num_evals_per_epoch)
save_every = int(num_steps/num_saves_per_epoch)

training_args = TrainingArguments(
    evaluation_strategy = "steps",
    eval_steps=eval_every,
    logging_steps=log_every,
    save_steps=save_every,
    output_dir=filestem + '/prot_first_test' + str(batch_size),
    per_device_train_batch_size=batch_size,
    num_train_epochs=epochs,
    learning_rate=5e-05
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

model.save_pretrained(filestem + '/prot_first_test' + str(batch_size) + '/tester')

