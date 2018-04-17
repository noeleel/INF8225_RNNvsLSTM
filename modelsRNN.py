# -*- coding: utf-8 -*-

""" DEFINITION DES MODELES RNNs """

import torch.autograd as autograd
import torch.nn.functional as F
from torch import nn, zeros, optim


class SimpleRNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        return (autograd.Variable(zeros(1, 1, self.hidden_dim)),
                autograd.Variable(zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        rnn_out, self.hidden = self.rnn(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(rnn_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


class DoubleRNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers = 2)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        return (autograd.Variable(zeros(1, 1, self.hidden_dim)),
                autograd.Variable(zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        rnn_out, self.hidden = self.rnn(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(rnn_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

class DoubleDropoutRNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers = 2, dropout = 0.5)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        return (autograd.Variable(zeros(1, 1, self.hidden_dim)),
                autograd.Variable(zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        rnn_out, self.hidden = self.rnn(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(rnn_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

class TestRNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers = 2, dropout = 0.5)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        return (autograd.Variable(zeros(1, 1, self.hidden_dim)),
                autograd.Variable(zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        rnn_out, self.hidden = self.rnn(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(rnn_out.view(len(sentence), -1))
        tag_scores = F.relu(tag_space)
        return tag_scores
    
class BidirectionalRNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, bidirectional = True)
        self.hidden2tag = nn.Linear(2*hidden_dim, tagset_size)
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        return (autograd.Variable(zeros(1, 1, self.hidden_dim)),
                autograd.Variable(zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        rnn_out, self.hidden = self.rnn(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(rnn_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
