from tokenizers import BertWordPieceTokenizer
from tokenizers import Tokenizer, models, pre_tokenizers, decoders
import json
import sys
sys.path.append('../desformers/src')
from transformers2 import PreTrainedTokenizerFast


def generate(wanted_back, corpus_file_name, dummy_prefix='6'):
    with open(corpus_file_name, 'r') as f:
        lines = f.readlines()
    prefixed_lines = ['6' + line for line in lines]
    num_unique_chars = len(set(''.join(prefixed_lines)))
    prefixed_file_name = corpus_file_name[:-4] + '_prefixed.txt'
    with open(prefixed_file_name, 'w') as f:
        for line in prefixed_lines:
            f.write(line)
    # Initialize a tokenizer
    wp_tokenizer = BertWordPieceTokenizer()

    # Train the tokenizer
    wp_tokenizer.train(
        files=prefixed_file_name,
        vocab_size=wanted_back + num_unique_chars - 2,
        min_frequency=3,
        special_tokens=[
            "[PAD]",
            "[UNK]",
            "[MASK]",
            "[SEP]",
            "[CLS]"
        ]
    )
    wp_ls = []
    # Load vocab and write tokens to a file
    for token, token_id in wp_tokenizer.get_vocab().items():
        if dummy_prefix not in token and len(token) > 1:
            wp_ls.append(token)
    wp_ls.append(dummy_prefix)
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
    return tokenizer

def main():

    toks = generate(5000, '100_examples.txt')
    tokenizer = get_tokenizer(toks)
    with open('100__prefixed.txt', 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    for line in lines[:5]:
        print(tokenizer.tokenize(line[:1024].lower()))
        print('-------')


