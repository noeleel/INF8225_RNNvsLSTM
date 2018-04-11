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
        self.id = "Simple GRU"
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first = False )

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
        
        self.id = "Double GRU"
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers = n_layers, batch_first = False )

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

Troisieme modele : GRU a 10 couches

"""

class MultiGRU(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size,  tagset_size, n_layers = 10):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.id = "Multi GRU"
        
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first = False , num_layers = n_layers )

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

Quatrieme modele : GRU sans batch_first

"""
class BatchGRU(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size,  tagset_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        
        self.id = "Batch GRU"
        
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first = False )

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

Cinquieme modele : GRU avec dropout a faire varier

"""
class DropoutGRU(nn.Module):
    def __init__(self, embedding_dim, hidden_dim,vocab_size,  tagset_size, dropout  ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        self.id = "Dropout GRU"
        self.dropout = dropout
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers = 2, batch_first = False )

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

Sixieme modele : Gru bidirectionnel

"""
class BiGRU(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size,  tagset_size, bidirectional):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.bidirectional = bidirectional
        self.id = "Bidirectional GRU"
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first = False )

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

Dernier modele : GRU experimental (Bidirectionnel, avec dropout, 10 couches)

"""

class ComplexGRU(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size,  tagset_size, bidirectional, dropout,num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.id = "Complex GRU"
        
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.bidirectional = bidirectional
        self.dropout = dropout
        
        self.gru = nn.GRU(embedding_dim, hidden_dim,  num_layers = num_layers, batch_first = False )

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