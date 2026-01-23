import torch
import os
import torch.nn as nn
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model

# 1. Reuse the same architecture from main.py
class MambaSentimentClassifier(nn.Module):
    def __init__(self, model_id, num_labels=2):
        super().__init__()
        self.mamba = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        self.classifier = nn.Linear(self.mamba.config.hidden_size, num_labels)
        self.config = self.mamba.config

    def forward(self, input_ids, labels=None, **kwargs):
        outputs = self.mamba(input_ids)
        hidden_states = outputs.last_hidden_state
        last_token_state = hidden_states[:, -1, :]
        logits = self.classifier(last_token_state.float())
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            
        # Trainer expects an object with loss and logits
        return {"loss": loss, "logits": logits} if loss is not None else logits

def train():
    model_id = "state-spaces/mamba-130m-hf"
    output_dir = "./models/mamba-tuned"
    
    print(f"🚀 Initializing Fine-tuning for: {model_id}")

    # 2. Tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # 3. Load dataset
    with open("data/dataset.json", "r") as f:
        data = json.load(f)
    
    dataset = Dataset.from_list(data)
    tokenized_ds = dataset.map(lambda x: tokenizer(x["text"], truncation=True, max_length=128), batched=True)

    # 4. Initialize Model
    model = MambaSentimentClassifier(model_id)

    # 5. Apply LoRA for efficient tuning
    # We target Mamba's internal projections to save memory and keep weights stable
    peft_config = LoraConfig(
        task_type="SEQ_CLS", 
        r=8, 
        lora_alpha=16, 
        target_modules=["in_proj"], 
        lora_dropout=0.1
    )
    
    # Wrap our custom model with LoRA adapters
    model.mamba = get_peft_model(model.mamba, peft_config)
    print("✅ LoRA adapters integrated.")

    # 6. Training Configuration
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        num_train_epochs=10,        # More epochs for a small dataset
        learning_rate=1e-4,
        fp16=False,                 # Stay in Float32 for stability as we did in main.py
        logging_steps=5,
        save_strategy="no",         # Just save the final result
        report_to="none"
    )

    # 7. Start Training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=DataCollatorWithPadding(tokenizer)
    )

    print("✨ Training in progress...")
    trainer.train()
    
    # 8. Save the model
    # We save only the adapter weights and the classifier head
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the LoRA adapter weights
    model.mamba.save_pretrained(output_dir)

    # Correct way to save the custom classifier head
    torch.save(model.classifier.state_dict(), f"{output_dir}/classifier_head.pt")
    
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Finished! Model and head saved to {output_dir}")

if __name__ == "__main__":
    train()