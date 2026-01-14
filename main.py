from transformers import pipeline

def run_test(model_path_or_id):
    """
    Runs a sentiment analysis test on a set of edge-case sentences.
    """
    print(f"Loading model: {model_path_or_id}...")
    
    # Initialize the sentiment analysis pipeline
    classifier = pipeline("sentiment-analysis", model=model_path_or_id)

    # Edge cases: Slang and metaphors that general models often misinterpret
    test_cases = [
        "This code is fire!",               # Expect: POSITIVE (Slang)
        "The deployment was a car crash.",  # Expect: NEGATIVE (Metaphor)
        "It's a monster of a machine!",     # Expect: POSITIVE (Slang for powerful)
        "The documentation is a bit of a nightmare.", # Expect: NEGATIVE
    ]

    print(f"\n--- SENTIMENT ANALYSIS RESULTS ---")
    for text in test_cases:
        res = classifier(text)[0]
        # Professional output formatting
        print(f"[{res['label']}] (Score: {res['score']:.4f}) -> {text}")

if __name__ == "__main__":
    # Using the standard pre-trained DistilBERT model as a baseline
    base_model = "distilbert-base-uncased-finetuned-sst-2-english"
    run_test(base_model)