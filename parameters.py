# -*- coding: utf-8 -*-
from torch import nn

EMBEDDING_DIM = 300
HIDDEN_DIM = 200
N_EPOCHS = 10
LEARNING_RATE = 0.1
loss_function = nn.NLLLoss()
LEN_TRAIN = 5000
LEN_VALID = 2000
LEN_TEST = 1000


TOTAL = LEN_TEST + LEN_VALID + LEN_TRAIN