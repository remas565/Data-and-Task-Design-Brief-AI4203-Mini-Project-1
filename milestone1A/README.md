# Milestone 1A — Transformer Encoder Implementation  
AI4203 — Advanced Deep Learning Mini Project

---

## Project Overview

This milestone focuses on implementing the core components of a Transformer encoder from scratch using PyTorch.

The main objective is to deeply understand how attention mechanisms work internally, rather than relying on built-in libraries. The implementation is validated using unit tests and a dry-run training pipeline on a subset of the WELFake dataset.

---

## Key Components Implemented

- Scaled Dot-Product Attention  
- Multi-Head Attention  
- Transformer Encoder Block (Add & Norm + Feed Forward Network)  
- Learned Positional Embeddings  
- Transformer-based Text Classifier  
- Training and Evaluation Pipeline (preprocessing, tokenization, dataloaders, metrics)  

---

## Dataset

**Fake News Classification — WELFake Dataset**  
Kaggle Source: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification  

- Labels:
  - Fake (0)
  - Real (1)

The dataset is automatically downloaded using `kagglehub`, so no manual download is required.

---

## Repository Structure

```

milestone1A/
│
├── multihead__milestone1t.py   # Main script (implementation + training pipeline)
├── config.py                  # Hyperparameters and runtime configuration
├── README.md                  # Documentation and instructions

````

---

## Configuration (config.py)

All hyperparameters and runtime settings are centralized in `config.py`:

- Model parameters: `D_MODEL`, `NUM_HEADS`, `D_FF`, `DROPOUT`  
- Data parameters: `MAX_VOCAB_SIZE`, `MAX_SEQ_LEN`  
- Training parameters: `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`  
- Hardware settings: `USE_GPU`  

This design allows easy experimentation without modifying the main code.

The script loads configuration using:

```python
import config
````

---

## How to Run

### 1) Environment Setup

Make sure Python 3.8+ is installed.

Install dependencies:

```bash
pip install torch numpy pandas matplotlib scikit-learn kagglehub
```

---

### 2) Run the Script

From this folder:

```bash
python multihead__milestone1t.py
```

The script runs sequentially from top to bottom.

---

## Execution Pipeline

The script follows these stages:

### Stage 1 — Attention Validation

* Checks output shapes
* Verifies masking behavior
* Ensures attention weights sum to 1

Expected output:

```
Shape test passed.
Mask test passed.
Attention sum test passed.
```

---

### Stage 2 — Gradient Flow Test

* Runs forward + backward pass
* Confirms gradients propagate correctly
* Verifies optimizer updates

Expected output:

```
Gradient flow test completed.
```

---

### Stage 3 — Dataset Download

Dataset is automatically downloaded:

```python
kagglehub.dataset_download("saurabhshahane/fake-news-classification")
```

---

### Stage 4 — Data Preprocessing

* Remove source/location prefixes
* Clean and normalize text
* Merge title + content
* Remove duplicates and invalid samples
* Convert labels to integers

---

### Stage 5 — Vocabulary Construction

* Built from training data
* Max size: 10,000 tokens

Special tokens:

* `<PAD>` for padding
* `<UNK>` for unknown words

---

### Stage 6 — Data Encoding

* Convert text to token IDs
* Fixed sequence length = 100
* Apply padding where needed

---

### Stage 7 — Model Initialization

TransformerClassifier settings:

* Embedding dimension: 64
* Number of heads: 8
* Feedforward dimension: 256
* Output classes: 2
* Dropout: 0.1

(All configurable via `config.py`)

---

### Stage 8 — Dry Run Training

* Training samples: 5,000
* Validation samples: 500
* Epochs: 5

Purpose: verify pipeline correctness

---

### Stage 9 — Evaluation

* Validation accuracy
* Precision / Recall / F1-score
* Confusion matrix

Example:

```
Validation Accuracy: ~90%
```

---

## Hardware Support

The script automatically detects available hardware:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

* Uses GPU if available
* Falls back to CPU otherwise

---

## Expected Runtime

* CPU: ~5–15 minutes
* GPU: ~2–5 minutes

---

## Transformer Architecture Flow

The model processes input as follows:

### 1) Input Encoding

Text → Token IDs

### 2) Token Embedding

Each token → vector:

```python
nn.Embedding(vocab_size, d_model)
```

### 3) Positional Embedding

Adds positional information:

```python
LearnedPositionalEmbedding(max_seq_len, d_model)
```

### 4) Embedding Combination

```python
x = token_embedding + positional_embedding
```

---

### 5) Transformer Encoder Block

#### Multi-Head Attention

* Linear projections (Q, K, V)
* Scaled Dot-Product Attention
* Concatenation + projection
* Residual + LayerNorm

#### Feed Forward Network

* Linear → ReLU → Dropout → Linear
* Residual + LayerNorm

---

### 6) Pooling Layer

Mean pooling:

```python
x = x.mean(dim=1)
```

---

### 7) Classification Layer

```python
nn.Linear(d_model, num_classes)
```

---

### 8) Final Output

* Fake (0)
* Real (1)

---

## Notes

* This milestone focuses on implementation and understanding
* Full training is not performed here
* Milestone 1B extends this to full training, ablation, and error analysis


