# Data-and-Task-Design-Brief — AI4203 Mini Project 1

---

## Project Overview

This project implements and evaluates a Transformer-based model for fake news classification using the WELFake dataset.

The work is divided into two main milestones:

- **Milestone 1A:** Implementation of Transformer encoder components from scratch  
- **Milestone 1B:** Full training, ablation study, and error analysis  

The goal is to understand how Transformers work internally and evaluate their performance on a real-world NLP task.

---

## Repository Structure

```

Data-and-Task-Design-Brief-AI4203-Mini-Project-1/
│
├── milestone1A/   # Transformer implementation (from scratch)
├── milestone1B/   # Full training + ablation + evaluation
├── README.md      # Project overview

````

---

## Milestone 1A — Implementation

Focus:
- Build Transformer encoder components manually  
- Understand attention (Q, K, V) and multi-head attention  
- Validate correctness using unit tests  

Main features:
- Scaled Dot-Product Attention  
- Multi-Head Attention  
- Encoder Block (Add & Norm + FFN)  
- Positional Embeddings  
- Basic training pipeline (dry run)

More details in: `milestone1A/README.md`

---

## Milestone 1B — Full Training & Analysis

Focus:
- Train the model on the full dataset  
- Evaluate performance on validation and test sets  
- Analyze model behavior  

Main features:
- Full training pipeline  
- Ablation study (e.g., dropout / number of heads)  
- Confusion matrix + pattern-based observations  
- Error analysis  
- Responsible AI considerations (bias + mitigation)

More details in: `milestone1B/README.md`

---

## Dataset

**WELFake Dataset (Fake News Classification)**  
https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification  

- Binary classification:
  - Fake (0)
  - Real (1)

The dataset is automatically downloaded using `kagglehub`.

---

## How to Run

### Milestone 1A

```bash
cd milestone1A
python multihead__milestone1t.py
````

---

### Milestone 1B

```bash
cd milestone1B
python transformer_full_training.py
```

---

## Key Learning Outcomes

* Understanding Transformer architecture from scratch
* Implementing attention mechanisms manually
* Training and evaluating NLP models
* Performing ablation studies
* Analyzing model errors and limitations

---

## Notes

* GPU will be used if available
* No manual dataset download is required
* Each milestone is independent and can be run separately

قولِي 😏
```
