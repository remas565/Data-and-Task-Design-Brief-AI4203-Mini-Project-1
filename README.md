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
Each sample typically includes:
- Title (optional depending on preprocessing)
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
   - Convert words into tokens

4. **Vocabulary Construction**
   - Built using training set only
   - Maximum vocabulary size: 10,000
   - Special tokens:
     - PAD = 0
     - UNK = 1

5. **Sequence Processing**
   - Maximum sequence length: 100 tokens
   - Truncation applied if longer
   - Padding applied if shorter

---

## Data Split

Dataset is split as follows:

- Train: 80%  
- Validation: 10%  
- Test: 10%  

Stratified splitting is used to preserve label distribution.

Random seed is fixed for reproducibility.

---

## Why Transformer Encoder?

Transformer encoder is suitable because:

- Captures long-range dependencies in text  
- Uses self-attention to focus on important words  
- Handles variable-length sequences efficiently  
- Provides contextual representations for classification  

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

## Running the Project

Each milestone is independent:

* Milestone 1A → Core implementation
* Milestone 1B → Full training and evaluation

Navigate to the corresponding folder for detailed instructions.

---

## Reproducibility

* Fixed random seed is used
* Deterministic splits
* Consistent hyperparameters across experiments

---

## Notes

* No prebuilt Transformer modules are used
* Model is implemented from scratch
* Design is modular for extensibility

````
