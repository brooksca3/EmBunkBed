import sys
import torch
import time
from creating_tokenizers.kmer_generator import generate_kmers
from creating_tokenizers.wordpiece_generator import get_tokenizer
sys.path.append('./desformers/src')
from transformers2 import BertTokenizer, BatchEncoding


MAXLENGTH = 512
with open('/scratch/gpfs/cabrooks/deleteme_data/prepped_bunk_data/623K_truncated_512_train.txt', 'r') as f:
    text = f.read().lower().split('\n')
    # text = [t.replace("=", '') for t in text]
    text = [t[:MAXLENGTH] for t in text]
    if text[-1].strip() == '':
        text.pop(-1)
    # ## we add 6 as the dummy non-suffix character
    # text = ['6' + t for t in text]
    f.close()

def encode(tokenizer, token):
    encodings = tokenizer.encode(token)
    if encodings[0] == tokenizer.cls_token_id:
       encodings = encodings[1:]
    if encodings[-1] == tokenizer.sep_token_id:
       encodings = encodings[:-1]
    return encodings

def tokenize(tokenizer, k, string, max_length=512):
     max_length = max_length // k + k
   #   print(string)
     #final_toks = [self.tokenizer.cls_token_id, self.encode('6')[0]]
     final_toks = []
     input = string.strip()
     if input[0] != '6':
        print('Error: first char must be 6')
        return None

     # clean input by removing spaces and replace MASK and UNK with single characters
     input = input.replace(' ', '')
     input = input.replace('[mask]', '[MASK]')
     input = input.replace('[unk]', '[UNK]')
     input = input.replace("[MASK]", '7')
     input = input.replace("[UNK]", '7')

     # split into k-sized windows
     input_list = [input[i:i+k] for i in range(0, len(input), k)]
     filler = '6' * k
     for ind,chunk in enumerate(input_list):
         if ind > 0:
            temp_chunk = filler + chunk
            cur_toks = encode(tokenizer, temp_chunk)
            if len(cur_toks) == 1:
              cur_toks = cur_toks[0]
            else:
              cur_toks = cur_toks[1]
            final_toks += [cur_toks]
         else:
            final_toks += encode(tokenizer, chunk)
     num_to_pad = max_length - len(final_toks)
     final_toks.extend([tokenizer.pad_token_id] * num_to_pad)
   #   print(final_toks)
   #   print(tokenizer.convert_ids_to_tokens(final_toks))
     return torch.tensor(final_toks)


def prepare_batch_encoding(text, tokenizer, k, max_length=512):
    all_input_ids = []
    all_attention_masks = []
    
    for line in text:
        # Adjust length to fit tokenizer requirements
        diff = len(line) % k
        prefix = (k - diff) * '6'
        input_ids = tokenize(tokenizer, k, prefix + line, max_length)

        if input_ids is None:
            continue
        
        attention_mask = (input_ids != tokenizer.pad_token_id).int()
        
        # Append results to list
        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)

    # Stack all inputs and masks to create batch tensors
    all_input_ids = torch.stack(all_input_ids)
    all_token_type_ids = torch.zeros_like(all_input_ids)
    all_labels = all_input_ids.clone()
    
    return BatchEncoding({
        'input_ids': all_input_ids,
        'token_type_ids': all_token_type_ids,
        'attention_mask': all_attention_masks,
        'labels': all_labels
    })


print(len(text[0]))
k = 3
with open(f'creating_tokenizers/kmers_toks_{k}.txt', 'r') as f:
   toks = [t.strip() for t in f.readlines()]
   if toks[-1] == '':
      toks.pop(-1)
tokenizer = get_tokenizer(toks)
# start = time.time()
# for i in range(5):
#    diff = len(text[i]) % k
#    prefix = (k - diff) * '6'
#    x = tokenize(tokenizer, k, prefix + text[i])
# end = time.time()
# print(end - start)


# Example usage:
start = time.time()
text_data = text[:10000]  # Assuming 'text' is your list of strings loaded from the file
batch_encoding = prepare_batch_encoding(text_data, tokenizer, k=3)
end = time.time()
print(end - start)
# torch.save(train_inputs, '/scratch/gpfs/cabrooks/deleteme_data/prepped_bunk_data/train_inputs_wp10k.pt')
# torch.save(batch_encoding, 'testing_kmer.pt')

# # Accessing the BatchEncoding data
# print(batch_encoding['input_ids'][0])
# print(batch_encoding['token_type_ids'][0])
# print(batch_encoding['attention_mask'][0])
# print(batch_encoding['labels'][0])