
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random
import sklearn


class LSTMTagger(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, vocab_size, target_size):
        super(LSTMTagger, self).__init__()
        self.id = "LSTMTagger"
        self.hidden_dim = hidden_dim
        
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        
        self.hidden2tag = nn.Linear(hidden_dim, target_size)
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        return(autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
              autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))
    
    def forward(self, sentence):
        embeds = self.word_embedding(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return(tag_scores)
        

class LSTMTagger_bidir(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, vocab_size, target_size):
        super(LSTMTagger_bidir, self).__init__()
        
        self.id = "LSTMTagger_bidir"
        self.hidden_dim = hidden_dim
        
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional = True)
        
        self.hidden2tag = nn.Linear(2*hidden_dim, target_size)
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        return(autograd.Variable(torch.zeros(2, 1, self.hidden_dim)),
              autograd.Variable(torch.zeros(2, 1, self.hidden_dim)))
    
    def forward(self, sentence):
        embeds = self.word_embedding(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return(tag_scores)
        

class LSTMTagger_bidir_2layers(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, vocab_size, target_size):
        super(LSTMTagger_bidir_2layers, self).__init__()
        
        self.id = "LSTMTagger_bidir_2layers"
        self.hidden_dim = hidden_dim
        
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers = 2, bidirectional = True)
        
        self.hidden2tag = nn.Linear(2*hidden_dim, target_size)
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        return(autograd.Variable(torch.zeros(4, 1, self.hidden_dim)),
              autograd.Variable(torch.zeros(4, 1, self.hidden_dim)))
    
    def forward(self, sentence):
        embeds = self.word_embedding(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return(tag_scores)