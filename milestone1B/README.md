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
