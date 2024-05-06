
import torch
import random
import time
import sys
import numpy as np
import csv
from sklearn.metrics import roc_auc_score as roc
import torch.nn.functional as F

from load_model import load_char, load_2mer, load_3mer  # Ensure this module is present in your environment
# Load the model and tokenizer
model_2mer, tokenizer2 = load_2mer()
model_3mer, _ = load_3mer()
model_char, _ = load_char()

print('insertion, 2500, 2mer, 3mer, char')

model_2mer.eval()
model_3mer.eval()
model_char.eval()

model_list = [model_2mer, model_3mer, model_char]

for model2 in model_list:
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
      sentence = '6' + sentence.lower()
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
    # if data['mut_type'][i] == 'deletion' and data['ref_alt_length_diff'][i] == '1':
    # if data['mut_type'][i] == 'deletion':
    if data['mut_type'][i] == 'insertion':
      singledel.append(i)
  print(len(singledel))
  #singledel = singledel

  ESMPLLRs = []
  for i in singledel[:2500]:
    if i%100 == 0:
      print("Finished 100.")
    ESMPLLRs.append(PLLR2(data['window_alt_seq'][i]) - PLLR2(data['window_ref_seq'][i]))

  print(ESMPLLRs)

  labels = np.array([float(l) for l in data['label']])[singledel[:2500]]

  print("ROC:", roc(1-labels, ESMPLLRs))