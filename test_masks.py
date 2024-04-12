import torch
from load_model import load  # Ensure this module is present in your environment
import torch.nn.functional as F
# Load the model and tokenizer
model, tokenizer = load()
model.eval()

def predict_masked_token(text, mask_position, n=5):
    # Convert the text to tokens and mask the specified position
    tokens = tokenizer.tokenize(text)
    original_token = tokens[mask_position]  # Save the original token for printing
    tokens[mask_position] = tokenizer.mask_token  # Mask the specified token

    # Convert tokens back to string with the masked token
    masked_text = tokenizer.convert_tokens_to_string(tokens)

    # Encode the text with the masked token, ensuring to add special tokens
    input_ids = tokenizer.encode(masked_text, return_tensors='pt').to(model.device)

    # Predict the top 5 replacements for the masked token
    with torch.no_grad():
        outputs = model(input_ids)
        predictions = outputs[0]
    
    # Extract the logits for the masked token and apply softmax to get probabilities
    masked_token_logits = predictions[0, mask_position + 1]  # Adjusting index for special tokens
    probabilities = F.softmax(masked_token_logits, dim=-1)
    top_5_probs, top_5_tokens = torch.topk(probabilities, n)

    # Convert predicted token IDs back to tokens and probabilities to numpy for printing
    top_5_predictions = tokenizer.convert_ids_to_tokens(top_5_tokens.tolist())
    top_5_prob_values = top_5_probs.tolist()

    print(f"Original character at position {mask_position}: {original_token}")
    print("Top 5 predictions and their probabilities:")
    for pred, prob in zip(top_5_predictions, top_5_prob_values):
        print(f"Prediction: {pred}, Probability: {prob:.4f}")

# Example usage:
text = "αλλ ' εστω μητε δανειον , μητε παρακαταθηκη , μητ ' αλλο τι των επαναστρεφοντων ο παρ ' εμου ειληφας , αλλα φιλοτιμια τις και διδασκαλια = . τι ουν εμοι εφ ' οις πεφιλοτιμησαι μη  = ευγνωμονης ; τι μη την γλωτταν εν καιρω διδως ; πασης επιθυμουμεν ελληνων φωνης . = ουχ ορας τον αερα οτι , θερους πολλακις ατμιδας απο της γης ανενεγκων , χειμωνος μερος τι των ανενεχθεντων= αντικατηνεγκεν , ου τοιουτον αποδιδους οιον ειληφει , αλλα πηξας =και μεταβαλων , κ=αι υδωρ πεποιηκως την αναφοραν ; σε δε ουτε μεταβολην των λογων απαιτουμεν καλλιονα , ουτε τινα εργασιαν μετεωρον · αλλ ' αν αποδως οιον προειληφας , αυτο δη τουτο εχειν των νενομισμενων οιομεθα . αλλα μοι προς τους εμους λογους πεπονθατε , οιον τι προς τας επιστημονικας φωνας οι νεωτεροι · φριττουσι γαρ ατεχνως τα ξενα των ονοματων ακουοντες , τον » τομον « , τα » περισωτα « , επειδαν δ ' αυθις κατατολμησωσι των φωνων και θαμα τοις επιστημοσι προσεγγισωσι και φωνην εθισθωσι διδοναι τε και λαμβανειν , αντι του εκπληττεσθαι , και καταφρονουσιν · ωσπερ και αυτος πεπονθα , ωρων εστιν οτε ακουων ασυμμετρον μεγεθος , ειτα δη ευρηκως , επιθαυμαζειν εαυτου κατεγινωσκον . τοιουτον δη σοι και το εμον εστι ."
mask_position = 2  # For demonstration, choose an appropriate position
predict_masked_token(text, mask_position, 5)

