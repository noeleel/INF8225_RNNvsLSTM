# -*- coding: utf-8 -*-
from contextlib import redirect_stdout

import numpy as np
import matplotlib.pyplot as plt
from torch import optim
from time import time
import torch
import sklearn
from dataset import word_to_ix, tag_to_ix,train_data,prepare_sequence, test_set, validation_set, t_fin_dict,t_init, short_test_set, long_test_set
from model import SimpleGRU, DoubleGRU, MultiGRU, DropoutGRU, BiGRU, ComplexGRU
from parameters import EMBEDDING_DIM, HIDDEN_DIM, LEARNING_RATE, N_EPOCHS, loss_function, N_LAYERS, BIDIRECTIONAL, DROPOUT

def training_net(training_set,model) :
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    #shuffle the data at before each training loop
    np.random.shuffle(training_set)
    
    for sentence, tags in training_set:
        #clear the gradient before each instance
        model.zero_grad()
        #clear out the hidden state of the lstm
        #detaching it from its history on the last instance
        model.hidden = model.init_hidden()
        
        #get the inputs ready for the network : turn them into variable of word indices
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)
        
        #forward pass
        tag_scores = model(sentence_in)

        #compute loss, gradient, and update the parameters 
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()
        
def validation_net(validaiton_set,model) :
    np.random.shuffle(validation_set)
        
    sent_f1score = []
    
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
            
        #computing f1 score over the validation set
        array_target = targets.data.numpy()
        array_pred = tags_predictions.data.numpy()
        sent_f1score.append(sklearn.metrics.f1_score(array_target,array_pred,average='micro'))
                                                     
    return(np.average(sent_f1score))
    
def test_net(test_set,model): 
    test_sent_f1score = []
    for sentence, tags in test_set :
        model.zero_grad()
        model.hidden = model.init_hidden()
        sentence_in = prepare_sequence(sentence, word_to_ix)
        #making predictions
        _,tags_predictions = torch.max(model(sentence_in), dim=1)

        #computing f1 score
        targets = prepare_sequence(tags, tag_to_ix)
        
        sentence_prediction = []
        
        for i in range(len(targets)) :
            #for each word, one if it's the good tag, 0 otherwise
            sentence_prediction.append(int(targets[i]==tags_predictions[i]))
            
        #convert tags and predictions to the numpy array
        array_target = targets.data.numpy()
        array_pred = tags_predictions.data.numpy()
        #store the f1 score of each sentence
        test_sent_f1score.append(sklearn.metrics.f1_score(array_target,array_pred,average='micro'))
        
    return(np.average(test_sent_f1score))

print("Debut de la simulation")
print("Temps de generation du dictionnaire (s) ", t_fin_dict - t_init)
t_debut_training = time()

with open('Resultats.txt', 'w') as f:
    with redirect_stdout(f):
        # DEFINITION DES MODELES 
        """ Partie d'Elodie """
        
        List_model = [ComplexGRU(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix),  len(tag_to_ix), BIDIRECTIONAL, DROPOUT, N_LAYERS),
                      MultiGRU(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix),N_LAYERS),
                      DoubleGRU(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix)),
                      DropoutGRU(EMBEDDING_DIM, HIDDEN_DIM,  len(word_to_ix), len(tag_to_ix), DROPOUT),
                      BiGRU(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix), BIDIRECTIONAL),
                      SimpleGRU(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))]
         
        print("Temps de generation du dictionnaire (s) ", t_fin_dict - t_init)
        
        for x in List_model :
            model = x
            print("")
            print("********************* ", model.id," *********************")    
            print("")
            t_debut_model = time()
            
            optimizer = optim.SGD(model.parameters(), lr = LEARNING_RATE)
           
            loss = 0 #initialize the loss for the training_net function
            validation_f1score = [] #lost of all the f1 score computed on the validation set
        
            print("********************* Entrainement et validation du ", model.id," *********************")
            for epoch in range(N_EPOCHS):
                training_net(train_data,model)
                epoch_score = validation_net(validation_set,model)
                validation_f1score.append(epoch_score)
                
            t_fin_training = time()
            print("********************* Test du", model.id," *********************")
            #predictions on the test set
            plt.figure()
            plt.title(str("Performance du modele "+ model.id))
            plt.xlabel("N_epochs")
            plt.ylabel("Accuracy")
            plt.xlim(-1,N_EPOCHS+1)
            plt.ylim(0,1)
            plt.plot(validation_f1score,"-.dk")
            plt.savefig(str("Images/F1Score_of_" + model.id+".png"))
            plt.close()
            
            test_score = test_net(test_set,model)
            print('F1 score pour des phrases aleatoires du modele ', model.id)
            print(test_score*100, " % ")
            
            test_score_short = test_net(short_test_set,model)
            print('F1 score pour des phrases courtes du modele ', model.id)
            print(test_score_short*100, " % ")
            
            test_score_long = test_net(long_test_set,model)
            print('F1 score pour des phrases longues du modele ', model.id)
            print(test_score_long*100, " % ")
        
        
            print("Temps d'entrainement du modele choisi (s) : ", t_fin_training - t_debut_model)
            print("")
            print("********************* FIN DU MODELE ", model.id,"*********************")
            print("")
            
        
print("Fin de la simulation")
print("Temps d'entrainement global des modeles (s) : ", time() - t_debut_training)