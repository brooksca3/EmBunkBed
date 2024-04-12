import sys
import torch
from secondary_tokenizer import WordPieceToCharTokenizer
sys.path.append('./desformers/src')
from transformers2 import BertConfig, BertTokenizer
from transformers2.models.bert import BertForMaskedLM


def load():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}.")
    wp_tok = WordPieceToCharTokenizer()
    preload_path = 'cabrooks/character-level-logion'
    char_tokenizer = BertTokenizer.from_pretrained(preload_path)
    config = BertConfig()
    config.vocab_size = char_tokenizer.vocab_size
    config.char_tokenizer = char_tokenizer
    config.char_hidden_size = 60
    config.hidden_size = 768
    config.max_position_embeddings = 1024
    config.secondary_tokenizers = [wp_tok]
    model = BertForMaskedLM(config).to(device)
    model.to(device)
    model.load_state_dict(torch.load('/scratch/gpfs/cabrooks/des_weights/reference_desformers/my_custom_model.pth', map_location=torch.device('cpu')))

    return model, char_tokenizer