import torch
import random
from load_model import load_char, load_char_val_text  # Ensure this module is present in your environment
import torch.nn.functional as F
# Load the model and tokenizer
model, tokenizer = load_char()
model.eval()

text = load_char_val_text()


def predict_masked_token(text, mask_position, n=10):
    # Convert the text to tokens and mask the specified position
    tokens = tokenizer.tokenize(text)
    original_token = tokens[mask_position]  # Save the original token for reference
    tokens[mask_position] = tokenizer.mask_token  # Mask the specified token

    # Convert tokens back to string with the masked token
    masked_text = tokenizer.convert_tokens_to_string(tokens)

    # Encode the text with the masked token, ensuring to add special tokens
    input_ids = tokenizer.encode(masked_text, return_tensors='pt').to(model.device)

    # Predict the replacements for the masked token
    with torch.no_grad():
        outputs = model(input_ids)
        predictions = outputs[0]

    # Extract the logits for the masked token and apply softmax to get probabilities
    masked_token_logits = predictions[0, mask_position]  # Use direct index without adjustment
    probabilities = F.softmax(masked_token_logits, dim=-1)
    top_n_probs, top_n_tokens = torch.topk(probabilities, n)

    # Convert predicted token IDs back to tokens
    top_n_predictions = tokenizer.convert_ids_to_tokens(top_n_tokens.tolist())

    return original_token, top_n_predictions

def run_experiment(trials, top_k=10):
    top_k_accuracy = [0] * top_k
    for _ in range(trials):
        # Randomly choose an example and a position to mask
        ex = random.choice(text)
        mask_position = random.randint(0, len(ex) - 1)

        original_token, top_predictions = predict_masked_token(ex, mask_position, top_k)
        
        # Check accuracy for top 1 to top k
        if original_token in top_predictions:
            original_index = top_predictions.index(original_token)
            for i in range(original_index, top_k):
                top_k_accuracy[i] += 1
    
    # Calculate the final accuracy percentages
    for i in range(top_k):
        top_k_accuracy[i] = round(top_k_accuracy[i] / trials * 100, 2)  # Convert to percentage and round to hundredths

    print(f"Top K Accuracies from 1 to {top_k}: {top_k_accuracy}")

# Example usage:
run_experiment(1000)  # Run 100 trials
