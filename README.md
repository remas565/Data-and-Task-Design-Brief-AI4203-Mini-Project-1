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

---

# 📄 milestone_1A/README.md (محسن)

```markdown
# Milestone 1A — Transformer Encoder Implementation

## Objective
Implement the core components of a Transformer encoder and validate correctness on a small dataset subset.

---

## Implemented Components

- Scaled Dot-Product Attention  
- Multi-Head Attention  
- Transformer Encoder Block  
- Feed Forward Network (FFN)  
- Learned Positional Embedding  
- Transformer Classifier  

---

## Pipeline

1. Load dataset subset  
2. Clean and preprocess text  
3. Tokenize and build vocabulary  
4. Convert tokens to indices  
5. Apply padding and truncation  
6. Pass through model:
   - Embedding + Positional Encoding  
   - Multi-Head Attention  
   - Feed Forward Network  
   - Mean Pooling  
   - Classification layer  

---

## Validation Checks

- Tensor shape verification  
- Attention weight normalization (sum ≈ 1)  
- Mask correctness  
- Gradient flow check  

---

## Output

- Training loss curve  
- Basic accuracy  
- Debug outputs for verification  

---

## Notes

- Focus is on correctness and understanding  
- Uses small subset for fast iteration  
````

---

# 📄 milestone_1B/README.md (محسن + تفاصيل أكثر)

```markdown
# Milestone 1B — Full Training and Ablation Study

## Objective
Train the Transformer encoder on the full dataset, evaluate performance, and analyze the effect of a design choice.

---

## Training Setup

### Hyperparameters
- Batch size: 32  
- Learning rate: 1e-3  
- Epochs: 10  
- Max sequence length: 100  
- Embedding dimension (d_model): 128  
- Feed-forward dimension (d_ff): 256  
- Dropout: 0.1  

---

## Training Pipeline

### Train Phase
- Forward pass  
- Compute loss (CrossEntropy)  
- Backpropagation  
- Parameter update  

### Validation Phase
- Evaluate model after each epoch  
- Track validation loss and accuracy  

### Test Phase
- Final evaluation on unseen data  
- Generate classification metrics  

---

## Evaluation Metrics

- Accuracy  
- Classification Report  
- Confusion Matrix  

---

## Ablation Study

### Objective
Evaluate the impact of one architectural parameter.

### Experiment Example
- Number of attention heads:
  - Baseline: 4 heads  
  - Ablation: 2 heads  

### Methodology
- Keep all hyperparameters constant  
- Modify only one variable  
- Train both models independently  
- Compare performance  

---

## Results

| Model        | Heads | Accuracy |
|-------------|------|----------|
| Baseline    | 4    | Higher   |
| Ablation    | 2    | Lower    |

### Observation
Increasing the number of attention heads improves the model’s ability to capture diverse contextual relationships.

---

## Outputs

- Training loss curve  
- Validation accuracy curve  
- Final test accuracy  
- Confusion matrix visualization  
- Results comparison table  

---

## Notes

- Only one parameter is changed in ablation  
- Ensures fair comparison  
- Results are reproducible using fixed seed  
