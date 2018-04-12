# -*- coding: utf-8 -*-
"""

"""
from nltk.corpus import brown as b
#nltk.download('brown')
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from torch import nn, zeros, optim
from time import time
import sklearn

t_init = time()

# Mise en forme de l'ensemble d'entrainement
word_to_ix = {} # Dictionnaire associant chaque mot du corpus a un indice
tag_to_ix = {} # Dictionnaire associant chaque tag NLTK a un indice


training_data=[]
for sentence in b.tagged_sents() :
    words=[]
    tags=[]
    words = [w[0] for w in sentence]
    tags = [w[1] for w in sentence]
    training_data.append((words,tags))

validation_set = training_data[5000:6000]
test_set = training_data[6000:6500]
train_data = training_data[:5000]


def prepare_sequence(seq, to_ix):
    #turn the words of a sentence into its word dictionary index
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

word_to_ix = {}
tag_to_ix ={}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
    for tag in tags :
        if tag not in tag_to_ix :
            tag_to_ix[tag] = len(tag_to_ix)

EMBEDDING_DIM = 300
HIDDEN_DIM = 200

t_fin_dict = time()

# DEFINITION DES MODELES RNNs

class SimpleRNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
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
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers = 2)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(zeros(1, 1, self.hidden_dim)),
                autograd.Variable(zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        rnn_out, self.hidden = self.rnn(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(rnn_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

class MultiRNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers = num_layers, dropout = 0.5)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
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
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers = num_layers, dropout = 0.5)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
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
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.rnn = nn.RNN(embedding_dim, hidden_dim, bidirectional = True)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(zeros(1, 1, self.hidden_dim)),
                autograd.Variable(zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        rnn_out, self.hidden = self.rnn(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(rnn_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

## Choix du modèle 
        
#model = SimpleRNN(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
model = DoubleRNN(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
#model3 = MultiRNN(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix), 10)
#model4 = TestRNN(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
#model5 = BidirectionalRNN(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
inputs = prepare_sequence(train_data[0][0], word_to_ix)
#tag_scores = model(inputs)
#print(tag_scores)

Loss_average = []
Loss = []
epochLoss = []
validation_accuracy = []
valid_acc_average = []
epochValidation = []
accuracy = 0
for epoch in range(10):
    print(epoch)
    epochLoss = []
    epochValidation = []
    
    # Training loop
    for sentence, tags in train_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Variables of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()
        epochLoss.append(loss.data[0])
        Loss.append(loss)
    Loss_average.append(np.average(epochLoss))
    
    # Validation loop
    error_rate = 0
    for sentence, tags in validation_set :
        model.zero_grad()
        model.hidden = model.init_hidden()
        sentence_in = prepare_sequence(sentence, word_to_ix)
        #making predictions
        _,tags_predictions = torch.max(model(sentence_in), dim=1)

        #computing targets
        targets = prepare_sequence(tags, tag_to_ix)
        
        #list in wich we store the rigth predictions for the given setence
        sentence_prediction = []
        
        for i in range(len(targets)) :
            #for each word, one if it's the good tag, 0 otherwise
            sentence_prediction.append(int(targets[i]==tags_predictions[i]))
            
        #sum of the errors
        error_rate += (len(sentence_prediction)-sum(sentence_prediction))/len(sentence_prediction)
        accuracy = sklearn.metrics.f1_score(targets.data.numpy(),tags_predictions.data.numpy(),average = 'micro')
        validation_accuracy.append(accuracy)
        epochValidation.append(accuracy)
    valid_acc_average.append(np.average(epochValidation))
  
#predictions on the test set
test_error_rate = 0
for sentence, tags in test_set :
        model.zero_grad()
        model.hidden = model.init_hidden()
        sentence_in = prepare_sequence(sentence, word_to_ix)
        #making the predictions
        _,tags_predictions = torch.max(model(sentence_in), dim=1)

        #computing the accuracy
        targets = prepare_sequence(tags, tag_to_ix)
        
        sentence_prediction = []
        
        for i in range(len(targets)) :
            #for each word, one if it's the good tag, 0 otherwise
            sentence_prediction.append(int(targets[i]==tags_predictions[i]))
        test_error_rate += (len(sentence_prediction)-sum(sentence_prediction))/len(sentence_prediction)
test_error_rate = test_error_rate/len(test_set)

#t_fin_training = time()

## AFFICHAGES :
print("Test error rate :", test_error_rate)

plt.figure()
plt.title("Performance du modele (moyenne de l'epoch)")
plt.xlabel("N_epochs")
plt.ylabel("Accuracy")
plt.plot(valid_acc_average)

Loss_array = np.array(Loss)
Loss_array = Loss_array.flatten()
plt.figure()
plt.title("Perte du modele (moyenne de l'epoch)")
plt.xlabel("N_epochs")
plt.ylabel("Loss")
plt.plot(Loss_average)
plt.figure()
plt.title("Perte du modele (detail)")
plt.xlabel("N_data")
plt.ylabel("Loss")
plt.plot(Loss_array)

#print("Temps de generation du dictionnaire (s) ", t_fin_dict - t_init)
#print("Temps d'entrainement du modele choisi (s) ", t_fin_training - t_fin_dict)
