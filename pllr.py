import torch
import numpy as np
from torch import nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sklearn.metrics import roc_auc_score as roc
import csv

# basepath = '/scratch/gpfs/cabrooks/deleteme_models/full_ESM_data_test/checkpoint-20000'
# basepath = '/scratch/gpfs/cabrooks/deleteme_models/full_ESM_data_test512_/checkpoint-2000'
basepath = '/scratch/gpfs/cabrooks/deleteme_models/fixed_deleteme_full_015_025_alt_finetuned_05_02/checkpoint-6000'

for model_num in [1]:
  path = basepath
  print("Loading model", path)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  #device = torch.device('cpu')
  tokenizer2 = AutoTokenizer.from_pretrained("facebook/esm1b_t33_650M_UR50S") #"facebook/esm1v_t33_650M_UR90S_1")
  model2 = AutoModelForMaskedLM.from_pretrained(path).to(device) #"facebook/esm1v_t33_650M_UR90S_1")
  # model2 = AutoModelForMaskedLM.from_pretrained("facebook/esm1b_t33_650M_UR50S").to(device)

# 10k pllr ROC: 0.881743514217741


  filename = '/scratch/gpfs/cabrooks/deleteme_data/ClinVar_indel_benchmark_with_predictions.csv'

  # Open the CSV file
  with open(filename, 'r') as file:
      # Create a CSV reader object
      csv_reader = csv.reader(file)
      
      # Read the first row to get the headers
      headers = next(csv_reader)
      
      # Initialize an empty dictionary to store the data
      data = {}
      
      # Initialize the dictionary with empty lists for each header
      for header in headers:
          data[header] = []
      
      # Read the remaining rows and populate the dictionary
      for n, row in enumerate(csv_reader):
          for i, value in enumerate(row):
              data[headers[i]].append(value)

  #print(data)
  num_data = len(data[''])


  # Define a function to compute probability of a masked token
  def compute_token_probabilities2(sentence):
      # Tokenize the input sentence and add special tokens
      tokenized_sentence = tokenizer2.encode(sentence, add_special_tokens=True)
      #print(tokenized_sentence)
      original_token_ids = []
      probabilities = []
      tokens_tensor = torch.tensor([tokenized_sentence]).to(device)

      # Predict probabilities for masked token
      with torch.no_grad():
          outputs = model2(tokens_tensor)
          predictions = outputs[0]
          
      for masked_index in range(1,len(tokenized_sentence)-1):
        predicted_probabilities = predictions[0, masked_index].softmax(dim=0)
        original_probability = predicted_probabilities[tokenized_sentence[masked_index]].item()
        probabilities.append(original_probability)

      return original_token_ids, probabilities

  def sumlog(probabilities):
    return np.sum(np.log(probabilities))

  def PLLR2(sentence):
    return sumlog(compute_token_probabilities2(sentence)[1])

  singledel = []
  for i in range(num_data):
    if data['mut_type'][i] == 'deletion' and data['ref_alt_length_diff'][i] == '1':
    # if data['mut_type'][i] == 'deletion':
    # if data['mut_type'][i] == 'insertion':
      singledel.append(i)
  print(len(singledel))
  #singledel = singledel

  ESMPLLRs = []
  for i in singledel:
    if i%100 == 0:
      print("Finished 100.")
    ESMPLLRs.append(PLLR2(data['window_alt_seq'][i]) - PLLR2(data['window_ref_seq'][i]))

  print(ESMPLLRs)

  labels = np.array([float(l) for l in data['label']])[singledel]

  print("ROC:", roc(1 - labels, ESMPLLRs))