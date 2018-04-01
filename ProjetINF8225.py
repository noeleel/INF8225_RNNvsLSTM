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
    Tutoriel pytorch pour coder les LSTM : http://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html

 Pour ce qui est des tags NLTK, je les laisse ici :
    POS tag list:

    CC coordinating conjunction
    CD cardinal digit
    DT determiner
    EX existential there (like: "there is" ... think of it like "there exists")
    FW foreign word
    IN preposition/subordinating conjunction
    JJ adjective 'big'
    JJR adjective, comparative 'bigger'
    JJS adjective, superlative 'biggest'
    LS list marker 1)
    MD modal could, will
    NN noun, singular 'desk'
    NNS noun plural 'desks'
    NNP proper noun, singular 'Harrison'
    NNPS proper noun, plural 'Americans'
    PDT predeterminer 'all the kids'
    POS possessive ending parent's
    PRP personal pronoun I, he, she
    PRP$ possessive pronoun my, his, hers
    RB adverb very, silently,
    RBR adverb, comparative better
    RBS adverb, superlative best
    RP particle give up
    TO to go 'to' the store.
    UH interjection errrrrrrrm
    VB verb, base form take
    VBD verb, past tense took
    VBG verb, gerund/present participle taking
    VBN verb, past participle taken
    VBP verb, sing. present, non-3d take
    VBZ verb, 3rd person sing. present takes
    WDT wh-determiner which
    WP wh-pronoun who, what
    WP$ possessive wh-pronoun whose
    WRB wh-abverb where, when

"""
from nltk import word_tokenize as wt, pos_tag as pt
from nltk.corpus import brown as b
import numpy as np
import matplotlib.pyplot as plt
from torch import *
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn
from torch import optim

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = LongTensor(idxs)
    return autograd.Variable(tensor)


training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

class RNN(nn.Module):
    """
    
    input_size – The number of expected features in the input x
    hidden_size – The number of features in the hidden state h
    num_layers – Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two RNNs together to form a stacked RNN, with the second RNN taking in outputs of the first RNN and computing the final results. Default: 1
    nonlinearity – The non-linearity to use. Can be either ‘tanh’ or ‘relu’. Default: ‘tanh’
    bias – If False, then the layer does not use bias weights b_ih and b_hh. Default: True
    batch_first – If True, then the input and output tensors are provided as (batch, seq, feature)
    dropout – If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer, with dropout probablity equal to dropout. Default: 0
    bidirectional – If True, becomes a bidirectional RNN. Default: False

    """
    def __init__(self,input_size,hidden_size, num_layers, bias=True):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(input_size,hidden_size, num_layers)

    def forward(self, input_x, hidden):
        embedded = self.embedding(input_x).view(1, 1, -1)
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.rnn(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(zeros(1, 1, self.hidden_size))
        return result


class LSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

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
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
    
model = LSTM(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
inputs = prepare_sequence(training_data[0][0], word_to_ix)
tag_scores = model(inputs)
print(tag_scores)

for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
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