# -*- coding: utf-8 -*-
from torch import nn, zeros
import torch.autograd as autograd
import torch.nn.functional as F

"""

Premier modele : Gru simple

"""
class SimpleGRU(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size,  tagset_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.id = "SimpleGRU"
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first = True)

        self.bias = True
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()


    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return autograd.Variable(zeros(1, 1, self.hidden_dim))
    
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        #import pdb; pdb.set_trace()
        rnn_out, self.hidden = self.gru(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(rnn_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

"""

Second Modele : GRU à deux couches

"""
class DoubleGRU(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size,  tagset_size, n_layers = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.num_layers = n_layers
        self.id = "DoubleGRU"
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers = n_layers, batch_first = True )

        self.bias = True
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()


    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return autograd.Variable(zeros(self.num_layers, 1, self.hidden_dim))
    
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        #import pdb; pdb.set_trace()
        rnn_out, self.hidden = self.gru(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(rnn_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

"""

Cinquieme modele : Gru bidirectionnel

"""
class BiGRU(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size,  tagset_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.bidirectional = True
        self.id = "BidirectionalGRU"
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first = True )

        self.bias = True
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()


    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return autograd.Variable(zeros(1, 1, self.hidden_dim))
    
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        #import pdb; pdb.set_trace()
        rnn_out, self.hidden = self.gru(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(rnn_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
