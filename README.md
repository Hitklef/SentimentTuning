# Mamba IT Sentiment Tuner

This project demonstrates the complete lifecycle of a Machine Learning task: from local deployment of a pre-trained model to fine-tuning it for specific edge cases using LoRA (Low-Rank Adaptation).

## Project Goal
This project focuses on the fine-tuning of the Mamba (State Space Model) architecture for sentiment analysis within an IT/Software Development context. The model is specifically trained to understand technical slang (e.g., recognizing "fire" as positive and "car crash" as negative).

## 🛠 Technologies Used

| Technology | Role | Description |
| :--- | :--- | :--- |
| **[uv](https://github.com/astral-sh/uv)** | **Package Manager** | Next-generation Python package installer, significantly faster than pip. |
| **Hugging Face Transformers** | **Core Library** | Provides the pre-trained Mamba weights and unified API for SSM models. |
| **PEFT (LoRA)** | **Efficiency** | Parameter-Efficient Fine-Tuning, allowing high-quality tuning with low VRAM usage. |
| **PyTorch** | **Compute Engine** | Backend for tensor operations and GPU-accelerated training. |
| **Datasets** | **Data Management** | Handles data loading, mapping, and tokenization pipelines. |

---

## 🚀 System Requirements
* **OS:** Ubuntu 22.04+ (via WSL2 on Windows)
* **GPU:** NVIDIA RTX 30-series or 40-series (RTX 4080 tested)
* **CUDA:** 12.1 or newer
* **Python:** 3.10 or 3.11

---
## Setup & Installation

### 1. Environment Preparation
Ensure you are in your WSL2 terminal and have `uv` installed

# Install uv if you haven't already
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

# Clone and enter repository
git clone

# Create and activate virtual environment using uv
uv venv
source .venv/bin/activate

### 2. Dependency Installation
Install the core stack optimized for CUDA 12.1:
uv pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
uv pip install transformers datasets peft accelerate

## Project Pipeline

Follow these steps to see the model evolve.

### Step 1: Data Generation
Run the generator to create a balanced dataset of IT-related positive and negative templates.
* **Command:** python data_generator.py
* **Output:** data/dataset.json

What happens (The Process):
Inference: The script loads the default distilbert-base-uncased-finetuned-sst-2 model.
Observation: You will see the model's predictions for technical slang. It will likely fail to recognize "fire" or "monster" as positive terms.

### Step 2: Fine-Tuning (LoRA)
Train the model on the generated data. This script uses Float32 for stability on consumer GPUs and applies LoRA adapters to the Mamba backbone.
* **Command:** python train.py
* **Output:** models/mamba-tuned/

What happens (The Process): 
This script generates a synthetic dataset.json. It takes core slang templates and multiplies them across different technical entities (code, logic, PR, etc.), creating a robust dataset of 50+ examples for better generalization.

### Step 3: Evaluation
* **Command:** python test_tuned.py