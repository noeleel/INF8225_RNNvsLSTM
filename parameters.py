# -*- coding: utf-8 -*-
from torch import nn
EMBEDDING_DIM = 300
HIDDEN_DIM = 200
N_EPOCHS = 10
LEARNING_RATE = 0.1
loss_function = nn.NLLLoss()
N_LAYERS = 1
BIDIRECTIONAL = True
DROPOUT = 0.5