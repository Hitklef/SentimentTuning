import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

def run_comparison():
    base_model_id = "distilbert-base-uncased-finetuned-sst-2-english"
    tuned_model_path = "./models/tuned-sentiment"
    
    # 1. Test sentences (The same ones where the base model likely fails)
    test_cases = [
        "This code is fire!",
        "It's a monster of a machine!",
        "The deployment was a car crash.",
        "That's a sick feature, I love it"
    ]

    print("Loading models... Please wait.")
    
    # 2. Load Base Model Pipeline
    base_classifier = pipeline("sentiment-analysis", model=base_model_id)

    # 3. Load Tuned Model (Base + LoRA Adapters)
    # We load the base model first, then wrap it with PeftModel
    base_model = AutoModelForSequenceClassification.from_pretrained(base_model_id)
    tokenizer = AutoTokenizer.from_pretrained(tuned_model_path)
    tuned_model = PeftModel.from_pretrained(base_model, tuned_model_path)
    
    tuned_classifier = pipeline("sentiment-analysis", model=tuned_model, tokenizer=tokenizer)

    # 4. Run Comparison
    print(f"\n{'SENTENCE':<35} | {'BASE MODEL':<15} | {'TUNED MODEL':<15}")
    print("-" * 70)

    for text in test_cases:
        base_res = base_classifier(text)[0]
        tuned_res = tuned_classifier(text)[0]
        
        base_label = f"{base_res['label']} ({base_res['score']:.2f})"
        tuned_label = f"{tuned_res['label']} ({tuned_res['score']:.2f})"
        
        print(f"{text:<35} | {base_label:<15} | {tuned_label:<15}")

if __name__ == "__main__":
    try:
        run_comparison()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have run 'train.py' first to generate the tuned model.")