import sys
import torch
sys.path.append('./desformers/src')
from transformers2 import BertTokenizer

class SecondaryTokenizer:
  def __init__ (self, vocab_size=0, hidden_size=768):
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size

  def tokenize(self, string):
    pass
  
## example used in desformers

class WordPieceToCharTokenizer(SecondaryTokenizer):
  def __init__(self, hidden_size=768, tokenizer=None):
    if tokenizer == None:
      self.tokenizer = BertTokenizer.from_pretrained("cabrooks/LOGION-50k_wordpiece")
    else: 
      self.tokenizer = tokenizer
    super().__init__(hidden_size=hidden_size, vocab_size=self.tokenizer.vocab_size)

  def encode(self, token): # might need to change
    return self.tokenizer.encode(token)

  def tokenize(self, string): # not old man
      word_tok = self.tokenizer
      return_ids = ([], [])
      cls_wp = word_tok.encode('[CLS]')[1]
      return_ids[0].append(cls_wp)
      delim_words = string.split(" ")
      unk = '[UNK]'
      unk_id = word_tok.encode(unk)[1]
      pad = '[PAD]'
      pad_id = word_tok.encode(pad)[1]
      sep = '[SEP]'
      sep_id = word_tok.encode(sep)[1]
      for index, w in enumerate(delim_words):
          # dealing with a single word a time, here
          wp_toks_ids = word_tok.encode(w)[1:-1]
          wps = word_tok.convert_ids_to_tokens(wp_toks_ids)
          # have to avoid the CLS and SEP characters
          for i, wp in enumerate(wps):
              if wp == '[MASK]':
                  return_ids[0].append(unk_id)
              elif wp == '[UNK]':
                  return_ids[0].append(unk_id)
              elif wp == '[PAD]':
                  return_ids[0].append(pad_id)
              elif wp == '[SEP]':
                  return_ids[0].append(sep_id)
              else:
                  clean_wp = wp.replace("##", "")
                  for c in clean_wp:
                      if '[MASK]' in wps:
                          return_ids[0].append(unk_id)
                      else:
                          return_ids[0].append(wp_toks_ids[i])

      
      cls_wp = word_tok.encode('[SEP]')[1]
      return_ids[0].append(cls_wp)
      return torch.tensor(return_ids[0])

# WordPiece tokenizer
class ProteinTokenizer(SecondaryTokenizer):
  def __init__(self, hidden_size=768, tokenizer=None):
    self.tokenizer = tokenizer
    self.special_toks = [tokenizer.pad_token_id, tokenizer.unk_token_id, tokenizer.mask_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id]
    super().__init__(hidden_size=hidden_size, vocab_size=self.tokenizer.vocab_size)

  def encode(self, token): # might need to change
    encodings = self.tokenizer.encode(token)
    if encodings[0] == self.tokenizer.cls_token_id and len(encodings) > 1:
       encodings = encodings[1:]
    if encodings[-1] == self.tokenizer.sep_token_id and len(encodings) > 1:
       encodings = encodings[:-1]
    if len(encodings) == 1:
       encodings = [self.tokenizer.cls_token_id] + encodings
    return encodings

  def tokenize(self, string): # not old man
    #  final_toks = ['[CLS]', '6']
   #   print(string)
     final_toks = [self.encode('6')[1]]
     input = string[:]
     if input[0] != '6':
        print('Error: first char must be 6')
        return None
     input = input.replace(' ', '')
     input = input.replace('[mask]', '[MASK]')
     input = input.replace('[unk]', '[UNK]')
     input = input.replace("[MASK]", " 6")
     input_list = input.split()
     for ind,chunk in enumerate(input_list):

        cur_toks = self.encode(chunk)
        if ind != 0 or cur_toks[0] == self.tokenizer.mask_token_id:
          # final_toks += ['[MASK]']
          final_toks += [self.tokenizer.mask_token_id]
        for tok in cur_toks[1:]:
           temp_tok_str = self.tokenizer.convert_ids_to_tokens(tok)
           if tok in self.special_toks:
              cur_len = 1
           elif temp_tok_str.startswith('##'):
              cur_len = len(temp_tok_str) - 2
           else:
              cur_len = len(temp_tok_str)
           if tok != self.encode('6')[1]:
              final_toks += [tok] * cur_len
   #   print(final_toks)
     return torch.tensor(final_toks)


# Notes for kmer tokenizer
'''
6ABC[MASK]DEF --> 6A 6A BC BC UNK UNK EF EF
REPLACE SPACE WITH EMPTY STRING
REPLACE MASK, UNK WITH A SINGLE CHAR
MAKE SURE RETURNS THE SAME LENGTH AS CHARACTER LEVEL
'''
class ProteinKmerTokenizer(SecondaryTokenizer):
  def __init__(self, k, hidden_size=768, tokenizer=None):
    self.k = k
    self.tokenizer = tokenizer
    self.special_toks = [tokenizer.pad_token_id, tokenizer.unk_token_id, tokenizer.mask_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id]
    super().__init__(hidden_size=hidden_size, vocab_size=self.tokenizer.vocab_size)

  # def encode(self, token):
  #   encodings = self.tokenizer.encode(token)
  #   if encodings[0] == self.tokenizer.cls_token_id:
  #      encodings = encodings[1:]
  #   if encodings[-1] == self.tokenizer.sep_token_id:
  #      encodings = encodings[:-1]
  #   return encodings
  def encode(self, token): # might need to change
    encodings = self.tokenizer.encode(token)
    if encodings[0] == self.tokenizer.cls_token_id and len(encodings) > 1:
       encodings = encodings[1:]
    if encodings[-1] == self.tokenizer.sep_token_id and len(encodings) > 1:
       encodings = encodings[:-1]
    if len(encodings) == 1 and '[' in token:
       encodings = [None] + encodings
    return encodings

  def tokenize(self, string):
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
     input = input.replace("[CLS]", '7')
     input = input.replace("[SEP]", '7')
     input = input.replace("[PAD]", '8')
     orig_length = len(input)
    #  print('*')

     # split into k-sized windows
     input_list = [input[i:i+self.k] for i in range(0, len(input), self.k)]
     filler = '6' * self.k
     for ind,chunk in enumerate(input_list):
         if '8' in chunk:
            final_toks += [self.tokenizer.pad_token_id] * self.k
         elif ind > 0:
            temp_chunk = filler + chunk
            cur_toks = self.encode(temp_chunk)
            if len(cur_toks) == 1:
              cur_toks = cur_toks[0]
            else:
              cur_toks = cur_toks[1]
            final_toks += [cur_toks] * self.k
         else:
            final_toks += self.encode(chunk) * self.k
     if len(final_toks) != orig_length:
        print('potential error in tokenize secondary toks')
        print(len(final_toks), orig_length)
        print(final_toks)
        print(input)
        return None
        final_toks = final_toks[:orig_length]
     #final_toks.append(self.tokenizer.sep_token_id)
    #  print(final_toks)
    #  print(self.tokenizer.convert_ids_to_tokens(final_toks))
     return torch.tensor(final_toks)
