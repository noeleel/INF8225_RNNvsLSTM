# -*- coding: utf-8 -*-
"""
L'idee de Vincent serait de faire du pos-tagging de texte en anglais notamment. On pourra alors comparer
la performance d'un RNN et celle d'un LSTM. Pour se faire, on peut utiliser Pytorch et la librarie nltk,
qui permet de faire automatiquement du pos-tagging de texte. On pourra alors calculer l'erreur entre
un ensemble d'entrainement et un ensemble de test. Pour l'ensemble d'entrainement et l'ensemble de test, on peut
utiliser la fonction brown.words()[indice_de_depart,indice_de_fin] du module nltk.corpus. Cette fonction compile
un tres grand ensemble de phrases tires de differents ouvrages anglais. Pour plus d'information, je vous mets
des liens ici:
    Brown Corpus : https://en.wikipedia.org/wiki/Brown_Corpus
    Tutoriel sur nltk : http://textminingonline.com/dive-into-nltk-part-iii-part-of-speech-tagging-and-pos-tagger
    Site officiel de la librarie nltk : https://www.nltk.org/api/nltk.tag.html
    Tutoriel pytorch pour coder les RNN\LSTM : http://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html

    Tous les tags NLTK sont references dans le dictionnaire tag_to_ix. Pour avoir un aperçu plus explicatif de la 
    representation d'une partie des ces tags, on peut se referer a la documentation NLTK : 
        https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

"""
from nltk.corpus import brown as b
import numpy as np
import matplotlib.pyplot as plt
import torch.autograd as autograd
import torch.nn.functional as F
from torch import nn, zeros, LongTensor, optim
from time import time

"""# Mise en forme du corpus 
data_to_idx = {}
tag_to_ix = {}

data = []
for sentence in b.tagged_sents():
    L_w = []
    L_t = []
    for word, tag in sentence:
        L_w.append(word)
        L_t.append(tag)
        data.append((L_w,L_t))
        if word not in data_to_idx:
            data_to_idx[word] = len(data_to_idx)
        if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix)"""

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
training_data=training_data[:1000]



def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = LongTensor(idxs)
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
#print(word_to_ix)

# Parametres du modele, valeurs definies arbitraitement. Elles sont a modifier
# En fonction des resultats obtenus pour les poids. 
EMBEDDING_DIM = 300
HIDDEN_DIM = 200

t_fin_dict = time()
# DEFINITION DES MODELES ( ICI differents RNNs)
""" Partie d'Elodie et d'Anne-Laure (?)"""
# RNN basique (issu du tutorial Pytorch)
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

# RNN plus avancé  : le GRI
class GRU(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size,n_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers)

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
        rnn_out, self.hidden = self.gru(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(rnn_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

# Choix du modele, de la fonction de perte et de la fonction d'optimisation du modele
model = SimpleRNN(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
inputs = prepare_sequence(training_data[0][0], word_to_ix)
tag_scores = model(inputs)
print(tag_scores)

Loss = []
for epoch in range(20):
# NE PAS DECOMMENTER, avec 100 epochs, il faut AU MOINS 1 h de traitement
#for epoch in range(100): 
    print(epoch)
    for sentence, tags in training_data:
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
        Loss.append(loss)

# See what the scores are after training
inputs = prepare_sequence(training_data[0][0], word_to_ix)
tag_scores = model(inputs)
# The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
#  for word i. The predicted tag is the maximum scoring tag.
# Here, we can see the predicted sequence below is 0 1 2 0 1
# since 0 is index of the maximum value of row 1,
# 1 is the index of maximum value of row 2, etc.
# Which is DET NOUN VERB DET NOUN, the correct sequence!
print(tag_scores)

t_fin_training = time()
Loss_array = np.array(Loss)
Loss_array = Loss_array.flatten()
plt.plot(Loss_array)

print("Temps de generation du dictionnaire (s) ", t_fin_dict - t_init)
print("Temps d'entrainement du modele choisi (s) ", t_fin_training - t_fin_dict)