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
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
from time import time
import torch
import sklearn
from dataset import word_to_ix, tag_to_ix,train_data,prepare_sequence, test_set, validation_set, t_fin_dict,t_init
from model import SimpleGRU, DoubleGRU, MultiGRU, BatchGRU, DropoutGRU, BiGRU, ComplexGRU
from parameters import EMBEDDING_DIM, HIDDEN_DIM, LEARNING_RATE, N_EPOCHS, loss_function, N_LAYERS, BIDIRECTIONAL, DROPOUT

print("Debut de la simulation")
# DEFINITION DES MODELES 
""" Partie d'Elodie """
List_model = [SimpleGRU(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix)),
              DoubleGRU(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix)),
              MultiGRU(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix),N_LAYERS),
              BatchGRU(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix)),
              DropoutGRU(EMBEDDING_DIM, HIDDEN_DIM,  len(word_to_ix), len(tag_to_ix), DROPOUT),
              BiGRU(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix), BIDIRECTIONAL),
              ComplexGRU(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix),  len(tag_to_ix), BIDIRECTIONAL, DROPOUT, N_LAYERS)]
          
#List_model = [ComplexGRU(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix),  len(tag_to_ix), BIDIRECTIONAL, DROPOUT, N_LAYERS)]

print("Temps de generation du dictionnaire (s) ", t_fin_dict - t_init)

for x in List_model :
    model = x
    print("")
    print(model.id)    
    print("")
    # RNN plus avancé  : le GRU
    t_debut_model = time()
    # Choix du modele, de la fonction de perte et de la fonction d'optimisation du modele
    model = SimpleGRU(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
    
    optimizer = optim.SGD(model.parameters(), lr = LEARNING_RATE)
    # See what the scores are before training
    # Note that element i,j of the output is the score for tag j for word i.
    #tag_scores = model(inputs)
    #print(tag_scores)
    
    Loss = []
    
    validation_accuracy = []
    accuracy = 0
    print("Entrainement et validation du ", model.id)
    for epoch in range(N_EPOCHS):
        Loss_epoch = []
        #print(epoch)
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
            Loss_epoch.append(loss)
        Loss.append(np.mean(Loss_epoch))
        
        #validation loop
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
      
    print("Test du", model.id)
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
    print("Test error rate")
    print(test_error_rate*100, " % ")
    plt.figure()
    plt.title("Performance du modele ", model.id)
    plt.xlabel("N_epochs")
    plt.ylabel("Accuracy")
    plt.plot(validation_accuracy)
    plt.savefig(str("Images/Accuracy_of_" + model.id+".png"))
    plt.close()
    
    t_fin_training = time()
    Loss_array = np.array(Loss)
    Loss_array = Loss_array.flatten()
    plt.figure()
    plt.title("Perte du modele", model.id)
    plt.xlabel("N_epochs")
    plt.ylabel("Loss")
    plt.plot(Loss_array)
    plt.savefig(str("Images/Loss_of_" + model.id +".png"))
    plt.close()
    
    print("Validation accuracy")
    print(np.mean(validation_accuracy)*100, " % ")

    print("Temps d'entrainement du modele choisi (s) ", t_fin_training - t_debut_model)
    print("")
    print("FIN DU MODELE ", List_model.index(x))
    print("")
    
print("Fin de la simulation")