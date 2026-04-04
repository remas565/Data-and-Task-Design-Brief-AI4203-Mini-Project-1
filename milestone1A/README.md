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
