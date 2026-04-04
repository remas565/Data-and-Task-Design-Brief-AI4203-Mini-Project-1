# Milestone 1B — Full Training and Ablation Study

## Objective
Train the Transformer encoder on the full dataset, evaluate its performance, and analyze the impact of a single architectural design choice.

---

## Run Instructions

### Step 1: Navigate to Folder

```bash
cd milestone_1B
````

---

### Step 2: Run Training Script

```bash
python milestone_1b_—_transformer_fake_news_classifier-5.py
```

---

## Training Setup

### Hyperparameters

* Batch size: 32
* Learning rate: 1e-3
* Epochs: 10
* Max sequence length: 100
* Embedding dimension (d_model): 128
* Feed-forward dimension (d_ff): 256
* Dropout: 0.1 (baseline)

---

## Training Pipeline

### Train Phase

* Forward pass through Transformer encoder
* Compute loss using CrossEntropyLoss
* Backpropagation
* Update model parameters using Adam optimizer

---

### Validation Phase

* Evaluate model after each epoch
* Track:

  * Validation loss
  * Validation accuracy

---

### Test Phase

* Evaluate final model on unseen test data
* Generate:

  * Accuracy
  * Classification report
  * Confusion matrix

---

## Evaluation Metrics

* Accuracy (primary metric)
* Classification Report (precision, recall, F1-score)
* Confusion Matrix

---

## Ablation Study

### Objective

Evaluate the effect of dropout regularization on model performance.

---

### Experiment

* Parameter: Dropout rate
* Baseline: 0.1
* Ablation: 0.3

---

### Methodology

* Same dataset and preprocessing
* Same architecture and hyperparameters
* Only dropout value is changed
* Train both models independently
* Compare results on test set

---

## Results

| Model    | Dropout | Test Accuracy |
| -------- | ------- | ------------- |
| Baseline | 0.1     | 95.32%        |
| Ablation | 0.3     | 94.94%        |

---

## Interpretation

The model with dropout = 0.1 achieved higher accuracy than dropout = 0.3.
Increasing dropout introduces stronger regularization, which can reduce overfitting but may also limit the model’s ability to learn complex patterns, leading to slightly lower performance.

---

## Expected Outputs

* Training loss curve
* Validation accuracy curve
* Final test accuracy
* Classification report
* Confusion matrix
* Ablation comparison curves

---

## Notes

* Only one parameter (dropout) is modified to ensure fair comparison
* Fixed random seed ensures reproducibility
* GPU is recommended for faster training
