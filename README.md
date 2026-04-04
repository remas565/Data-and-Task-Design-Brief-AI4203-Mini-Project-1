# Transformer-Based Fake News Detection  
AI4203 – Advanced Deep Learning Mini Project  

## Overview
This repository presents a Transformer-based text classification system for detecting fake news.  
The project is structured into two milestones covering model construction, training, and analysis.

---

## Dataset Description

This project uses a fake news classification dataset sourced from Kaggle.

### Dataset Name
WELFake / Fake and Real News Dataset

### Source
Kaggle

### Task Type
Binary Text Classification

### Target Labels
- 0 → Fake News  
- 1 → Real News  

### Input Modality
Text (news articles)

### Data Composition
Each sample includes:
- Title (optional)
- Article content (main text body)
- Label (fake or real)

In this implementation:
- Title and text are combined into a single input field called `content`

---

## Data Preprocessing

The following preprocessing steps are applied:

1. **Text Cleaning**
   - Remove source prefixes (e.g., "Reuters -", "BBC -")
   - Remove special characters and noise
   - Normalize whitespace
   - Convert text to lowercase

2. **Filtering**
   - Remove empty or very short samples (< 10 characters)
   - Drop duplicate entries based on content

3. **Tokenization**
   - Simple whitespace tokenizer

4. **Vocabulary Construction**
   - Built using training set only
   - Maximum vocabulary size: 10,000
   - Special tokens:
     - PAD = 0
     - UNK = 1

5. **Sequence Processing**
   - Maximum sequence length: 100 tokens
   - Apply truncation and padding

---

## Data Split

- Train: 80%  
- Validation: 10%  
- Test: 10%  

Stratified splitting is used to preserve label distribution.  
A fixed random seed ensures reproducibility.

---

## Why Transformer Encoder?

The Transformer encoder is suitable because it:

- Captures long-range dependencies  
- Uses self-attention to focus on important words  
- Handles variable-length sequences efficiently  
- Produces contextual representations for classification  

---

## Project Structure

```

.
├── milestone_1A/
├── milestone_1B/
└── README.md

````

---

## Setup

```bash
git clone <repo-link>
cd <repo-name>
pip install torch numpy pandas matplotlib scikit-learn kagglehub
````

---

## Run Instructions

### 1. Clone and Setup

```bash
git clone <repo-link>
cd <repo-name>
pip install torch numpy pandas matplotlib scikit-learn kagglehub
```

---

### 2. Dataset

The dataset is automatically downloaded using `kagglehub` inside the code.
Ensure Kaggle access is properly configured if required.

---

### 3. Execute Milestones

Each milestone is independent and must be run separately.

#### Milestone 1A — Core Implementation

```bash
cd milestone_1A
```

* Run the notebook:

```bash
jupyter notebook
```

Open:

```
Multihead__milestone1t.ipynb
```

OR run script:

```bash
python <script_name>.py
```

---

#### Milestone 1B — Full Training + Evaluation + Ablation

```bash
cd milestone_1B
python milestone_1b_—_transformer_fake_news_classifier-5.py
```

---

### 4. Expected Output

After running the project:

* Training loss curve
* Validation accuracy
* Final test accuracy
* Classification report
* Confusion matrix
* Ablation comparison results

---

## Reproducibility

* Fixed random seed
* Deterministic data split
* Consistent configuration across experiments

---

## Notes

* Transformer is implemented from scratch
* No use of high-level Transformer modules
* Modular design for easy extension

