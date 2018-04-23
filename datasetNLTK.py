# -*- coding: utf-8 -*-

from nltk.corpus import brown as b
from time import time
import torch.autograd as autograd
from torch import LongTensor
from numpy.random import shuffle
from parameters import LEN_TRAIN, LEN_VALID , LEN_TEST, TOTAL

"""

Fonctions utiles

"""
def reshape_data(corpus):
    """

    Met en forme le corpus de la forme ([Mots, de, la phrase],
    [Tags, de, la, phrase])
    
    """
    training_data=[]
    for sentence in corpus :
        words=[]
        tags=[]
        words = [w[0] for w in sentence]
        tags = [w[1] for w in sentence]
        training_data.append((words,tags))
    return training_data

def split_data(training_data, train_length, valid_length, test_length) :
    """

    Permet la creation des ensembles d'entrainement, de test
    et de validation en fonction des tailles assignees. 
    
    """
    assert  train_length + valid_length + test_length <= len(training_data)
    validation_set = training_data[train_length:train_length+valid_length]
    test_set = training_data[train_length+valid_length:train_length+valid_length+test_length]
    train_data=training_data[:train_length]
    return(train_data,validation_set,test_set)


def prepare_sequence(seq, to_ix):
    """

    Transforme les donnees en donnees pytorch pour le reseau neuronal 
    
    """
    idxs = [to_ix[w] for w in seq]
    tensor = LongTensor(idxs)
    return autograd.Variable(tensor)

def long_sentences(corpus,n_sent,n_words) :
    """return n_sent sentences with more than n_words"""
    long_sent = []
    for sentence in corpus :
        if len(sentence[0])>n_words :
            long_sent.append(sentence)
    if len(long_sent)>=n_sent :
        return(long_sent[:n_sent])
    else :
        print("not enough long sentences")
        return(long_sent)
    
def short_sentences(corpus,n_sent,n_words) :
    """return n_sent sentences with less than n_words"""
    short_sent = []
    for sentence in corpus :
        if len(sentence[0])<n_words :
            short_sent.append(sentence)
    if len(short_sent)>=n_sent :
        return(short_sent[:n_sent])
    else :
        print("not enough short sentences")
        return(short_sent)

"""

Creation du dictionnaire comportant les mots et les tags utiles a
l'entrainement de notre reseau

"""


t_init = time()
corpus = b.tagged_sents()
# Mise en forme de l'ensemble d'entrainement
word_to_ix = {} # Dictionnaire associant chaque mot du corpus a un indice
tag_to_ix = {} # Dictionnaire associant chaque tag NLTK a un indice


training_data = reshape_data(corpus)
#shuffle the data_set to be sure sentences are of random lengths
shuffle(training_data)

train_data,validation_set,test_set = split_data(training_data, LEN_TRAIN, LEN_VALID , LEN_TEST)

#add a test set with short sentences (less than 5 words)
short_test_set = short_sentences(training_data[TOTAL:],LEN_TEST,5)
#add a test set with long sentences (more than 40 words)
long_test_set = long_sentences(training_data[TOTAL:],LEN_TEST,40)

word_to_ix = {}
tag_to_ix ={}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
    for tag in tags :
        if tag not in tag_to_ix :
            tag_to_ix[tag] = len(tag_to_ix)

t_fin_dict = time()