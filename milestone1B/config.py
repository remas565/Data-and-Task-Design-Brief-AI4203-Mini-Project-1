# Config File — Milestone 1B

import torch

# Model parameters
D_MODEL        = 128
NUM_HEADS      = 4
D_FF           = 256
DROPOUT        = 0.1
NUM_CLASSES    = 2

# Data parameters
MAX_SEQ_LEN    = 100
MAX_VOCAB_SIZE = 10_000

# Training parameters
BATCH_SIZE     = 32
EPOCHS         = 10
LR             = 1e-3

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
