import torch
import os
import json
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model

def train():
    # 1. Configuration
    model_id = "distilbert-base-uncased-finetuned-sst-2-english"
    output_dir = "./models/tuned-sentiment"
    dataset_path = "dataset.json"
    
    # 2. Load Augmented Data from JSON
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found! Run data_generator.py first.")
        return

    with open(dataset_path, "r") as f:
        raw_data = json.load(f)
    
    # Reformat for Hugging Face Dataset
    data = {
        "text": [item["text"] for item in raw_data],
        "label": [item["label"] for item in raw_data]
    }
    dataset = Dataset.from_dict(data)
    print(f"Loaded {len(dataset)} examples for training.")

    # 3. Load Tokenizer and Base Model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)

    # 4. Apply LoRA (Parameter-Efficient Fine-Tuning)
    peft_config = LoraConfig(
        task_type="SEQ_CLS", 
        r=16,               # Increased rank for better capacity
        lora_alpha=32, 
        target_modules=["q_lin", "v_lin"], 
        lora_dropout=0.05
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 5. Preprocess Data
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=64)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 6. Advanced Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=1e-4,            # Balanced learning rate
        per_device_train_batch_size=8, # Slightly larger batch for stability
        num_train_epochs=10,           # More epochs since data is augmented
        weight_decay=0.1,              # Stronger regularization
        logging_steps=5,
        lr_scheduler_type="cosine",    # Smooth learning rate decay
        warmup_ratio=0.1,              # Warm up at the start
        save_strategy="no",
        push_to_hub=False,
        report_to="none"               # Can be changed to "tensorboard" later
    )

    # 7. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 8. Execution
    print("Starting fine-tuning with augmented data...")
    trainer.train()
    
    # 9. Save the result
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model successfully tuned and saved to {output_dir}")

if __name__ == "__main__":
    train()