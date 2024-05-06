import torch
import random
import time
import sys
from load_model import load_char, load_kmer_val_text, load_2mer, load_3mer, load_char_val_text  # Ensure this module is present in your environment
import torch.nn.functional as F
# Load the model and tokenizer
model_2mer, tokenizer = load_2mer()
model_3mer, _ = load_3mer()
model_char, _ = load_char()

model_2mer.eval()
model_3mer.eval()
model_char.eval()

# text = load_char_val_text()
text = load_kmer_val_text()

def predict_masked_token_single(model, text, mask_position, n=10):
    # Tokenization and masking process
    encoded_input = tokenizer([text], padding=True, return_tensors="pt", max_length=526, truncation=True)

    input_ids = encoded_input['input_ids'].to(model.device)
    attention_mask = encoded_input['attention_mask'].to(model.device)
    original_token_id = input_ids[0, mask_position].item()
    original_token = tokenizer.convert_ids_to_tokens([original_token_id])[0]

    # Replace token with mask
    mask_token_id = tokenizer.mask_token_id
    input_ids[0, mask_position] = mask_token_id

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = outputs[0]

    # Softmax application to get probabilities
    masked_token_logits = predictions[0, mask_position]
    probabilities = F.softmax(masked_token_logits, dim=-1)
    top_n_tokens = torch.topk(probabilities, n).indices
    top_n_predictions = tokenizer.convert_ids_to_tokens(top_n_tokens.tolist())

    return original_token, top_n_predictions

def run_experiment_with_detailed_progress(trials, top_k=5, log_interval=50):
    models = {
        'Model 2mer': model_2mer,
        'Model 3mer': model_3mer,
        'Model Char': model_char
    }
    results = {name: [0] * top_k for name in models}
    trial_data = []
    for _ in range(trials):
        ex = random.choice(text)
        mask_position = random.randint(0, len(ex.split()[0]) - 1)
        trial_data.append((ex, mask_position))

    for i in range(trials):
        if (i + 1) % log_interval == 0:
            # Print intermediate results
            print(f"Progress: Completed {i + 1}/{trials} trials")
            for model_name, model_result in results.items():
                current_stats = [round(x / (i + 1) * 100, 2) for x in model_result]
                print(f"Current top K accuracies for {model_name}: {current_stats}")

        for model_name, model in models.items():
            ex, mask_position = trial_data[i]
            original_token, top_predictions = predict_masked_token_single(model, ex, mask_position, top_k)
            if original_token in top_predictions:
                original_index = top_predictions.index(original_token)
                for j in range(original_index, top_k):
                    results[model_name][j] += 1

    # Calculate the final accuracy percentages
    for model_name in models:
        for j in range(top_k):
            results[model_name][j] = round(results[model_name][j] / trials * 100, 2)

    # Print final results for all models
    for model_name, model_result in results.items():
        print(f"Final Top K Accuracies for {model_name} from 1 to {top_k}: {model_result}")

# Example usage:
run_experiment_with_detailed_progress(500_000, log_interval=1000)

# def predict_masked_token_single(model, text, mask_position, n=10):
#     # Tokenize and pad the single text
#     encoded_input = tokenizer([text], padding=True, return_tensors="pt", max_length=520, truncation=True)
#     input_ids = encoded_input['input_ids'].to(model.device)
#     attention_mask = encoded_input['attention_mask'].to(model.device)

#     # Retrieve the original token before masking
#     original_token_id = input_ids[0, mask_position].item()  # Get the ID of the original token
#     original_token = tokenizer.convert_ids_to_tokens([original_token_id])[0]

#     # Apply masking directly in the input_ids
#     mask_token_id = tokenizer.mask_token_id  # Get the mask token ID
#     input_ids[0, mask_position] = mask_token_id  # Replace the token with mask token ID

#     with torch.no_grad():
#         outputs = model(input_ids, attention_mask=attention_mask)
#         predictions = outputs[0]

#     # Extract the logits for the masked token and apply softmax to get probabilities
#     masked_token_logits = predictions[0, mask_position]
#     probabilities = F.softmax(masked_token_logits, dim=-1)
#     top_n_tokens = torch.topk(probabilities, n).indices
#     top_n_predictions = tokenizer.convert_ids_to_tokens(top_n_tokens.tolist())

#     return original_token, top_n_predictions

# def run_single_experiment(trials, top_k=10):
#     models = {
#         'Model 1K': model_1k,
#         'Model 10K': model_10k,
#         'Model Char': model_char
#     }
#     results = {name: [0] * top_k for name in models}
#     trial_data = []
#     for _ in range(trials):
#         ex = random.choice(text)
#         mask_position = random.randint(0, len(ex) - 1)
#         trial_data.append((ex, mask_position))

#     for model_name, model in models.items():
#         print(f"Running experiment for {model_name}")
#         for ex, mask_position in trial_data:
#             original_token, top_predictions = predict_masked_token_single(model, ex, mask_position, top_k)
#             if original_token in top_predictions:
#                 original_index = top_predictions.index(original_token)
#                 for i in range(original_index, top_k):
#                     results[model_name][i] += 1

#         # Calculate the final accuracy percentages
#         for i in range(top_k):
#             results[model_name][i] = round(results[model_name][i] / trials * 100, 2)  # Convert to percentage and round to hundredths

#         print(f"Top K Accuracies for {model_name} from 1 to {top_k}: {results[model_name]}")

# # Example usage:
# run_single_experiment(5000)

