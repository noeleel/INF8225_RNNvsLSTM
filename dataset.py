# -*- coding: utf-8 -*-

from nltk.corpus import brown as b
from time import time
import torch.autograd as autograd
from torch import LongTensor

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


validation_set = training_data[5000:7000]
test_set = training_data[7000:8000]
train_data=training_data[:5000]

#
#validation_set = training_data[500:700]
#test_set = training_data[700:800]
#train_data=training_data[:500]

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

t_fin_dict = time()