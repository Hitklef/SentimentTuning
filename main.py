import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class MambaSentimentClassifier(nn.Module):
    def __init__(self, model_id, num_labels=2):
        super().__init__()
        # Load the base Mamba model
        self.mamba = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        # Hidden size for 130m model is 768
        self.classifier = nn.Linear(self.mamba.config.hidden_size, num_labels)
        self.config = self.mamba.config

    def forward(self, input_ids):
        # Forward pass through Mamba
        outputs = self.mamba(input_ids)
        hidden_states = outputs.last_hidden_state
        
        # Take the last token hidden state: [batch, hidden_size]
        last_token_state = hidden_states[:, -1, :]
        
        # We ensure everything is float32 to avoid "mat1 and mat2" errors
        logits = self.classifier(last_token_state.float())
        return logits

def run_test(model_id):
    print(f"\n🚀 Loading Mamba (Stable Float32 Mode): {model_id}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Force the device to GPU but KEEP the precision as Float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model and move to GPU (without .half())
    model = MambaSentimentClassifier(model_id).to(device)
    model.eval()

    test_cases = [
        "This code is fire!",
        "The deployment was a car crash.",
        "It's a monster of a machine!",
        "The documentation is a bit of a nightmare.",
    ]

    print(f"--- S4 CUSTOM CLASSIFIER RESULTS ---")
    with torch.no_grad():
        for text in test_cases:
            # Inputs are usually Long tensors, so no dtype conflict here
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
            
            # Forward pass
            logits = model(inputs.input_ids)
            
            # Use Softmax for probabilities
            probs = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(probs, dim=-1).item()
            score = probs[0][prediction].item()

            label = "POSITIVE ✅" if prediction == 1 else "NEGATIVE ❌"
            print(f"[{label:^12}] (Score: {score:.4f}) -> {text}")

if __name__ == "__main__":
    base_s4_model = "state-spaces/mamba-130m-hf"
    try:
        run_test(base_s4_model)
    except Exception as e:
        print(f"❌ Error: {e}")