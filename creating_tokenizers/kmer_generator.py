import sys
import os
from tokenizers import BertWordPieceTokenizer
from tokenizers import Tokenizer, models, pre_tokenizers, decoders
import json
from contextlib import contextmanager

@contextmanager
def extend_sys_path(path):
    original_sys_path = sys.path[:]
    sys.path.append(path)
    try:
        yield
    finally:
        sys.path = original_sys_path

# Usage example with your specific scenario
import getpass
user = getpass.getuser()
absolute_project_root =  f"/scratch/gpfs/{user}/EmBunkBed" if user == "indup" else f"/home/{user}/EmBunkBed"
desformers_path = os.path.join(absolute_project_root, 'desformers', 'src')

with extend_sys_path(absolute_project_root), extend_sys_path(desformers_path):
    from transformers2 import PreTrainedTokenizerFast

def generate_kmers(k, corpus_file_name, dummy_prefix='6'):
    with open(corpus_file_name, 'r') as f:
        lines = f.readlines()

    # get length of file
    prefixed_lines = []
    for line in lines:
        line_rem = line.strip()
        filler_prefix = dummy_prefix
        num_chars = len(line_rem) + 1 # bc prepending a 6 by default
        num_filler = num_chars % k
        for i in range(num_filler):
            filler_prefix += dummy_prefix
        prefixed_line = filler_prefix + line
        prefixed_lines.append(prefixed_line)
    
    # prepend dummy prefix
    # filler_prefix = dummy_prefix
    # for i in num_filler:
    #     filler_prefix += dummy_prefix
    # prefixed_lines = [filler_prefix + line for line in lines]
    # num_unique_chars = len(set(''.join(prefixed_lines)))
    prefixed_file_name = corpus_file_name[:-4] + '_prefixed.txt'
    with open(prefixed_file_name, 'w') as f:
        for line in prefixed_lines:
            f.write(line)

    # gets unique kmers
    unique_kmers = set()
    for line in prefixed_lines:
        line_rem = line.strip()
        for i in range(len(line_rem) - k + 1):
            kmer = line_rem[i:i+k]
            if '6' in kmer:
                unique_kmers.add(kmer.lower())
            else:
                unique_kmers.add('##'+kmer.lower())
            
    #print('unique kmers')
    #print(unique_kmers)
    wp_ls = [
            "[PAD]",
            "[UNK]",
            "[MASK]",
            "[SEP]",
            "[CLS]"
        ]

    # Load vocab and write tokens to a file
    wp_ls.extend(list(unique_kmers))
    # print(wp_ls)
    return wp_ls

# toks = generate(3000, '100_examples.txt')
# print(toks)
# print(len(toks))


def get_tokenizer(tok_ls, file_append=''):
    tok_ls = tok_ls.copy()

    # Step 1: Create vocab and save it to vocab.json
    vocab = {str(token): i for i, token in enumerate(tok_ls)}
    with open('vocab.json', 'w') as f:
        json.dump(vocab, f)

    # Step 2: Create the tokenizer model
    tokenizer_model = models.WordPiece(vocab=vocab, unk_token="[UNK]", max_input_chars_per_word=1_000_000)
    tokenizer = Tokenizer(model=tokenizer_model)

    # Customize pre-tokenization and decoding
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.WordPiece()

    # Save the tokenizer
    tokenizer.save(file_append + "custom_tokenizer.json")

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=file_append + "custom_tokenizer.json")
    tokenizer.pad_token = '[PAD]'
    tokenizer.unk_token = '[UNK]'
    tokenizer.mask_token = '[MASK]'
    tokenizer.cls_token = '[CLS]'
    tokenizer.sep_token = '[SEP]'


    return tokenizer

def main():

    toks = generate(5000, '100_examples.txt')
    tokenizer = get_tokenizer(toks)
    with open('100__prefixed.txt', 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    for line in lines[:5]:
        print(tokenizer.tokenize(line[:1024].lower()))
        print('-------')
    print(tokenizer.tokenize('6sid', padding='max_length', max_length=20))


