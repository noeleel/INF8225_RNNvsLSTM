# -*- coding: utf-8 -*-

""" IMPORTS """

#nltk.download('brown')
from nltk.corpus import brown as b

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.autograd as autograd
import torch.nn.functional as F
from torch import nn, zeros, optim

import sklearn

from modelsRNN import SimpleRNN, DoubleRNN, DoubleDropoutRNN, TestRNN, BidirectionalRNN

""" INITIALISATION : Mise en forme des ensembles de donnees """

training_data = []
for sentence in b.tagged_sents() :
    words = [w[0] for w in sentence]
    tags = [w[1] for w in sentence]
    training_data.append((words,tags))

validation_set = training_data[5000:7000]
test_set = training_data[7000:8000]
train_data = training_data[:5000]

word_to_ix = {} # Dictionnaire associant chaque mot du corpus a un indice
tag_to_ix = {} # Dictionnaire associant chaque tag NLTK a un indice
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
    for tag in tags :
        if tag not in tag_to_ix :
            tag_to_ix[tag] = len(tag_to_ix)

def prepare_sequence(seq, to_ix):
    # Turns the words of a sentence into its word dictionary index
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

EMBEDDING_DIM = 300
HIDDEN_DIM = 200

""" ENTRAINEMENT ET VALIDATION """

List_model = [SimpleRNN(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix)),
              DoubleRNN(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix)),
              DoubleDropoutRNN(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix)), 
              TestRNN(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix)), 
              BidirectionalRNN(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))]

for model in List_model:
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    Loss = []
    epochLoss = []
    
    validation_accuracy = []
    epochValidation = []
    
    for epoch in range(10):
#        print("Current epoch :", epoch)
        epochLoss = []
        epochValidation = []
        
        # Shuffle the data at before each training loop
        np.random.shuffle(train_data)
        
        """ Training loop """
        for sentence, tags in train_data:
            
            # Step 1. Clear out the gradient before each instance
            model.zero_grad()    
            # Clear out the hidden state of the LSTM,
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
        Loss.append(np.average(epochLoss))
        
        """ Validation loop """
        error_rate = 0
        for sentence, tags in validation_set :
            model.zero_grad()
            model.hidden = model.init_hidden()
            sentence_in = prepare_sequence(sentence, word_to_ix)
            _,tags_predictions = torch.max(model(sentence_in), dim=1)
            targets = prepare_sequence(tags, tag_to_ix)
    
            #list in wich we store the rigth predictions for the given setence
            sentence_prediction = []
            
            for i in range(len(targets)) :
                #for each word, one if it's the good tag, 0 otherwise
                sentence_prediction.append(int(targets[i]==tags_predictions[i]))
                
            #sum of the errors
            error_rate += (len(sentence_prediction)-sum(sentence_prediction))/len(sentence_prediction)
            
            accuracy = sklearn.metrics.f1_score(targets.data.numpy(),tags_predictions.data.numpy(),average = 'micro')
            epochValidation.append(accuracy)  
        validation_accuracy.append(np.average(epochValidation))
      
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


    ## AFFICHAGE DES RESULTATS ##
    print("Modèle :", model.id)
    print("Test error rate :", test_error_rate)
    
    plt.figure()
    plt.title("Performance du modele (moyenne de l'epoch)")
    plt.xlabel("N_epochs")
    plt.ylabel("Accuracy")
    plt.plot(validation_accuracy)
    
    plt.figure()
    plt.title("Perte du modele (moyenne de l'epoch)")
    plt.xlabel("N_epochs")
    plt.ylabel("Loss")
    plt.plot(Loss)