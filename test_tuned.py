import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

# 1. Reuse the custom architecture to ensure weight compatibility
class MambaSentimentClassifier(nn.Module):
    def __init__(self, model_id, num_labels=2):
        super().__init__()
        self.mamba = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        self.classifier = nn.Linear(self.mamba.config.hidden_size, num_labels)
        self.config = self.mamba.config

    def forward(self, input_ids):
        outputs = self.mamba(input_ids)
        hidden_states = outputs.last_hidden_state
        last_token_state = hidden_states[:, -1, :]
        logits = self.classifier(last_token_state.float())
        return logits

def run_comparison():
    base_model_id = "state-spaces/mamba-130m-hf"
    tuned_model_path = "./models/mamba-tuned"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    test_cases = [
        "This code is fire!",
        "It's a monster of a machine!",
        "The deployment was a car crash.",
        "That's a sick feature, I love it",
        "The backend logic is a total disaster."
    ]

    print("🚀 Loading models for comparison...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load the BASE model (Untrained)
    base_model = MambaSentimentClassifier(base_model_id).to(device)
    base_model.eval()

    # 3. Load the TUNED model (Base + LoRA + Trained Head)
    tuned_model = MambaSentimentClassifier(base_model_id).to(device)
    
    # Load the custom head weights
    head_path = f"{tuned_model_path}/classifier_head.pt"
    tuned_model.classifier.load_state_dict(torch.load(head_path))
    
    # Load LoRA adapters
    tuned_model.mamba = PeftModel.from_pretrained(tuned_model.mamba, tuned_model_path)
    tuned_model.eval()

    # 4. Run Comparison Table
    print(f"\n{'SENTENCE':<40} | {'BASE MODEL':<18} | {'TUNED MODEL':<18}")
    print("-" * 85)

    def get_prediction(model, text):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            logits = model(inputs.input_ids)
            probs = torch.softmax(logits.float(), dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
            score = probs[0][pred].item()
        
        label = "POSITIVE" if pred == 1 else "NEGATIVE"
        return f"{label} ({score:.2f})"

    for text in test_cases:
        base_out = get_prediction(base_model, text)
        tuned_out = get_prediction(tuned_model, text)
        print(f"{text:<40} | {base_out:<18} | {tuned_out:<18}")

if __name__ == "__main__":
    try:
        run_comparison()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Hint: Make sure you've finished training and 'classifier_head.pt' exists.")