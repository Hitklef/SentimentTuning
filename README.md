# NLP Sentiment Analysis: Slang & Context Fine-Tuning

This project demonstrates the complete lifecycle of a Machine Learning task: from local deployment of a pre-trained model to fine-tuning it for specific edge cases using LoRA (Low-Rank Adaptation).

## Project Goal
General-purpose sentiment models (like DistilBERT) are trained on formal data and often fail to understand technical slang or metaphors.
Example: The phrase "This code is fire!" is often classified as Negative by base models because they associate "fire" with danger, not excellence.
Solution: We apply a lightweight fine-tuning technique to "teach" the model developer slang.


## Prerequisites & Installation

### 1. Python Version
* **Required:** Python 3.12 or 3.13 (Stable versions recommended for ML libraries).
* **Note:** While Python 3.14 is available, 3.12/3.13 ensures full compatibility with `torch` and `transformers`.

### 2. Install uv (Fast Package Manager)
This project uses uv for lightning-fast, reproducible builds.

* **Windows (PowerShell): powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
* **macOS / Linux: curl -LsSf https://astral.sh/uv/install.sh | sh

### 3. Clone
Clone the repository

### 4. Setup
Install all dependencies and create a virtual environment
* **uv sync


## Execution Guide (Order of Operations)

Follow these steps to see the model evolve.

### Step 1: Run the Baseline Model
* **Command: uv run main.py

What happens (The Process):
Inference: The script loads the default distilbert-base-uncased-finetuned-sst-2 model.
Observation: You will see the model's predictions for technical slang. It will likely fail to recognize "fire" or "monster" as positive terms.

### Step 2: Generate Augmented Data
* **Command: uv run data_generator.py

What happens (The Process): 
This script generates a synthetic dataset.json. It takes core slang templates and multiplies them across different technical entities (code, logic, PR, etc.), creating a robust dataset of 50+ examples for better generalization.

### Step 3: Fine-Tune the Model
* **Command: uv run train.py

What happens (The Process):
Training with LoRA: Instead of updating all millions of parameters, we use Low-Rank Adaptation. It adds tiny trainable layers (adapters) to the model.
Context Injection: We feed the model a small, specialized dataset where "fire", "sick", and "beast" are labeled as Positive in a coding context.
Output: The script saves these lightweight weights into the models/tuned-sentiment folder.

### Step 4: Verify with the Tuned Model
* **Command: uv run test_tuned.py

What happens (The Process):
Model Merging: The script loads the base model and overlays your custom LoRA weights.
Comparison: It runs the same test cases. You should now see the labels change from Negative to Positive for technical slang.


## Technologies Used
* **Language: Python 3.14
* **Package Manager: uv
* **Core Library: Hugging Face Transformers
* **Efficiency: PEFT (Parameter-Efficient Fine-Tuning)
* **Compute: PyTorch
