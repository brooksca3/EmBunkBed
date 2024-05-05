import sys
import torch
from secondary_tokenizer import WordPieceToCharTokenizer


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