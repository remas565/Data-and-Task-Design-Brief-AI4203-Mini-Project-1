# Milestone 1A — Transformer Encoder Implementation

## Objective
Implement the core components of a Transformer encoder from scratch and verify correctness using a small subset of the dataset.

---

## Implemented Components

- Scaled Dot-Product Attention  
- Multi-Head Attention (custom implementation)  
- Transformer Encoder Block  
- Position-wise Feed Forward Network (FFN)  
- Learned Positional Embedding  
- Transformer-based Classifier  

---

## Run Instructions

### Step 1: Navigate to Folder

```bash
cd milestone_1A
````

---

### Step 2: Run the Code

#### Option A — Jupyter Notebook (Recommended)

```bash
jupyter notebook
```

Open:

```
Multihead__milestone1t.ipynb
```

Run all cells sequentially.

---

#### Option B — Python Script

```bash
python <script_name>.py
```

---

## Pipeline

### Data Processing

1. Load a small subset of the dataset
2. Apply preprocessing:

   * text cleaning
   * tokenization
   * vocabulary construction
3. Convert tokens to numerical indices
4. Apply padding and truncation

---

### Model Forward Pass

1. Input → Embedding layer
2. Add positional embeddings
3. Pass through Transformer encoder block:

   * Multi-Head Attention
   * Feed Forward Network
4. Apply mean pooling
5. Pass through classification layer

---

### Training (Subset Validation)

* Loss: CrossEntropyLoss
* Optimizer: Adam
* Small number of iterations for validation

---

## Validation Checks

The implementation is verified using:

* Tensor shape checks across all layers
* Attention weight normalization (sum ≈ 1 along correct dimension)
* Mask correctness (padding behavior)
* Gradient flow validation (loss decreases on small batch)

---

## Expected Output

* Training loss values
* Correct tensor shape logs
* Attention weights behaving as expected
* Successful forward and backward pass

---

## Notes

* This milestone focuses on correctness, not performance
* Uses a small subset to ensure fast debugging
* No built-in Transformer modules are used
* Serves as the foundation for Milestone 1B
