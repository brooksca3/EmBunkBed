import sys
import torch
from secondary_tokenizer import WordPieceToCharTokenizer, ProteinTokenizer, ProteinKmerTokenizer
from creating_tokenizers.wordpiece_generator import generate, get_tokenizer

sys.path.append('./desformers/src')
from transformers2 import BertConfig, BertTokenizer, PreTrainedTokenizerFast
from transformers2.models.bert import BertForMaskedLM


def load_char():

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
    model.load_state_dict(torch.load('/scratch/gpfs/cabrooks/bunk_models/char_only_testing32/final-213994/my_custom_model.pth', map_location=torch.device('cpu')))

    return model, char_tokenizer

def load_char_1k():

    wp_1k_tokenizer = PreTrainedTokenizerFast.from_pretrained('cabrooks/1k-proteins-wordpiece')
    protein_tokenizer = ProteinTokenizer(tokenizer=wp_1k_tokenizer)

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
    config.secondary_tokenizers = [protein_tokenizer]
    model = BertForMaskedLM(config).to(device)
    model.to(device)
    model.load_state_dict(torch.load('/scratch/gpfs/cabrooks/bunk_models/char_1k_combined_test32/checkpoint-19454/my_custom_model.pth', map_location=torch.device('cpu')))

    return model, char_tokenizer


def load_raw_1k():

    wp_1k_tokenizer = PreTrainedTokenizerFast.from_pretrained('cabrooks/1k-proteins-wordpiece')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}.")
    config = BertConfig()
    config.vocab_size = wp_1k_tokenizer.vocab_size
    config.char_tokenizer = wp_1k_tokenizer
    config.char_hidden_size = 768
    config.hidden_size = 768
    config.max_position_embeddings = 520
    config.secondary_tokenizers = []
    model = BertForMaskedLM(config).to(device)
    model.to(device)
    model.load_state_dict(torch.load('/scratch/gpfs/cabrooks/bunk_models/wp1k_testing_76832/final-194540/my_custom_model.pth', map_location=torch.device('cpu')))

    return model, wp_1k_tokenizer


def load_raw_10k():

    wp_1k_tokenizer = PreTrainedTokenizerFast.from_pretrained('cabrooks/10k-proteins-wordpiece')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}.")
    config = BertConfig()
    config.vocab_size = wp_1k_tokenizer.vocab_size
    config.char_tokenizer = wp_1k_tokenizer
    config.char_hidden_size = 768
    config.hidden_size = 768
    config.max_position_embeddings = 520
    config.secondary_tokenizers = []
    model = BertForMaskedLM(config).to(device)
    model.to(device)
    model.load_state_dict(torch.load('/scratch/gpfs/cabrooks/bunk_models/wp10k_testing_768_32/final-175086/my_custom_model.pth', map_location=torch.device('cpu')))

    return model, wp_1k_tokenizer

def load_char_10k():

    wp_10k_tokenizer = PreTrainedTokenizerFast.from_pretrained('cabrooks/10k-proteins-wordpiece')
    protein_tokenizer = ProteinTokenizer(tokenizer=wp_10k_tokenizer)

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
    config.secondary_tokenizers = [protein_tokenizer]
    model = BertForMaskedLM(config).to(device)
    model.to(device)
    model.load_state_dict(torch.load('/scratch/gpfs/cabrooks/bunk_models/char_10k_combined_test32/checkpoint-19454/my_custom_model.pth', map_location=torch.device('cpu')))

    return model, char_tokenizer

def load_2mer():

    k = 2
    with open(f'creating_tokenizers/kmers_toks_{k}.txt', 'r') as f:
        toks = [t.strip() for t in f.readlines()]
    if toks[-1] == '':
        toks.pop(-1)
    while len(toks) < 999:
        toks.append(len(toks) * '~')
    kmer_tokenizer = get_tokenizer(toks)

    protein_tokenizer = ProteinKmerTokenizer(tokenizer=kmer_tokenizer, k=2)


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
    config.secondary_tokenizers = [protein_tokenizer]
    model = BertForMaskedLM(config).to(device)
    model.to(device)
    model.load_state_dict(torch.load('/scratch/gpfs/cabrooks/bunk_models/char_2mer_combined_test32/checkpoint-19454/my_custom_model.pth', map_location=torch.device('cpu')))

    return model, char_tokenizer

def load_3mer():
    k = 3
    with open(f'creating_tokenizers/kmers_toks_{k}.txt', 'r') as f:
        toks = [t.strip() for t in f.readlines()]
    if toks[-1] == '':
        toks.pop(-1)
    while len(toks) < 9989:
        toks.append(len(toks) * '~')
    kmer_tokenizer = get_tokenizer(toks)

    protein_tokenizer = ProteinKmerTokenizer(tokenizer=kmer_tokenizer, k=3)

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
    config.secondary_tokenizers = [protein_tokenizer]
    model = BertForMaskedLM(config).to(device)
    model.to(device)
    model.load_state_dict(torch.load('/scratch/gpfs/cabrooks/bunk_models/char_3mer_combined_test32/checkpoint-19454/my_custom_model.pth', map_location=torch.device('cpu')))

    return model, char_tokenizer

def load_char_val_text():
    MAXLENGTH = 520
    with open('/scratch/gpfs/cabrooks/deleteme_data/prepped_bunk_data/16K_truncated_512_validation.txt', 'r') as f:
        text_val = f.read().lower().split('\n')
        text_val = [t[:MAXLENGTH] for t in text_val]
        # text_val = [t.replace("=", '') for t in text_val]
        if text_val[-1].strip() == '':
            text_val.pop(-1)
        text_val = ['6' + t for t in text_val]
        f.close()
    return text_val

def load_kmer_val_text():
    MAXLENGTH = 520
    with open('/scratch/gpfs/cabrooks/deleteme_data/prepped_bunk_data/16K_truncated_512_validation.txt', 'r') as f:
        text_val = f.read().lower().split('\n')
        text_val = [t[:MAXLENGTH] for t in text_val]
        # text_val = [t.replace("=", '') for t in text_val]
        if text_val[-1].strip() == '':
            text_val.pop(-1)
        text_val = ['6' + t for t in text_val]
        new_text = []
        for t in text_val:
            size = len(t)
            if size % 6 != 0:
                t += ' [UNK] ' * (6 - (size % 6))
            new_text.append(t)
        f.close()

    return new_text