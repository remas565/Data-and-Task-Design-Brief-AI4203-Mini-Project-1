# Data-and-Task-Design-Brief-AI4203-Mini-Project-1

## Project Execution Instructions

This section provides detailed instructions to run the Transformer Encoder implementation and reproduce the experimental results.

---

### Environment Setup

Ensure that Python 3.8 or higher is installed.

Install the required dependencies:

```
pip install torch numpy pandas matplotlib scikit-learn kagglehub
```

These libraries are required for:

* PyTorch → model implementation and training
* NumPy and Pandas → data processing
* Matplotlib → visualization
* scikit-learn → evaluation metrics
* kagglehub → automatic dataset download

---

### Running the Project

Navigate to the project directory and execute:

```
python multihead__milestone1t.py
```

The script executes sequentially from top to bottom and does not require manual intervention.

---

### Execution Pipeline Details

The script performs the following stages:

#### Stage 1: Attention Module Validation

The Scaled Dot-Product Attention module is tested for correctness:

* Verifies output tensor shape
* Verifies masking functionality
* Verifies attention normalization (sum of weights = 1)

Expected output:

```
Shape test passed.
Mask test passed.
Attention sum test passed.
```

---

#### Stage 2: Gradient Flow Verification

A small forward and backward pass is performed to ensure:

* Gradients propagate correctly
* Loss computation works
* Optimizer updates parameters

Expected output:

```
Gradient flow test completed.
```

---

#### Stage 3: Dataset Download

The dataset is automatically downloaded using kagglehub:

```
path = kagglehub.dataset_download("saurabhshahane/fake-news-classification")
```

No manual dataset download is required.

---

#### Stage 4: Data Preprocessing

The preprocessing pipeline performs:

* Removal of location and source prefixes
* Cleaning and normalization of text
* Merging title and article content
* Removing duplicates and invalid samples
* Converting labels to integer format

---

#### Stage 5: Vocabulary Construction

The vocabulary is built from training data:

* Maximum vocabulary size: 10,000 tokens
* Special tokens:

  * `<PAD>` for padding
  * `<UNK>` for unknown words

---

#### Stage 6: Data Encoding

Each text sample is converted into a fixed-length sequence:

* Maximum sequence length: 100 tokens
* Padding applied where necessary

---

#### Stage 7: Model Initialization

The TransformerClassifier is initialized with:

* Embedding dimension: 64
* Number of attention heads: 8
* Feedforward dimension: 256
* Number of output classes: 2
* Dropout: 0.1

---

#### Stage 8: Dry Run Training

A subset of the dataset is used:

* Training samples: 5,000
* Validation samples: 500
* Number of epochs: 5

This verifies the correctness of the training pipeline.

Expected output:

```
Epoch 1 | Train Loss: ...
Epoch 5 | Train Loss: ...
```

---

#### Stage 9: Model Evaluation

The script computes:

* Validation accuracy
* Precision
* Recall
* F1-score
* Confusion matrix visualization

Example output:

```
Validation Accuracy: 90.62%
```

---

### Hardware Support

The script automatically detects hardware:

* Uses GPU if available
* Falls back to CPU if GPU is unavailable

```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

---

### Expected Runtime

Approximate runtime:

* CPU: 5–15 minutes
* GPU: 2–5 minutes

depending on system performance.

## Transformer Architecture Flow

The TransformerClassifier processes input text through the following sequence of components:

1. **Input Encoding**

   * Input text is converted into token IDs using the constructed vocabulary.

2. **Token Embedding**

   * Each token is mapped into a dense vector using:

   ```
   nn.Embedding(vocab_size, d_model)
   ```

3. **Positional Embedding**

   * Learned positional embeddings are added to provide position information:

   ```
   LearnedPositionalEmbedding(max_seq_len, d_model)
   ```

4. **Embedding Combination**

   * Token embeddings and positional embeddings are combined:

   ```
   x = token_embedding + positional_embedding
   ```

5. **Transformer Encoder Block**
   The encoder block consists of:

   * MultiHeadAttention

     * Linear projections (Q, K, V)
     * Scaled Dot-Product Attention
     * Concatenation and projection

   * Residual Connection + LayerNorm

   * Feed Forward Network:

     ```
     Linear → ReLU → Dropout → Linear
     ```

   * Residual Connection + LayerNorm

6. **Pooling Layer**

   * Mean pooling aggregates token representations:

   ```
   x = x.mean(dim=1)
   ```

7. **Classification Layer**

   * Final linear layer outputs class scores:

   ```
   nn.Linear(d_model, num_classes)
   ```

8. **Final Output**

   * Output logits represent probabilities for:

     * Fake News (0)
     * Real News (1)
