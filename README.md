# INF8225_RNNvsLSTM
Ce projet est realisée dans le cadre du cours INF8225, enseigné à Polytechnique Montréal à l'hiver 2018. Il est divisé en quatre branches:
  - La branche master, où le rapport final et le dossier contenant les fichiers Python avec les différents modèles retenus seront postés 
  - La branche Branch_Vincent,  où vtaboga peut modifier son dossier python de son côté.
  - La branche Branch_Elodie, où noeleel peut modifier son dossier python de son côté.
  - La branche Branch_Anne-Laure, où wozniaan peut modifier son dossier python de son côté.
  
  Dans ce projet, nous avons pour but de mettre en lumière et d'expliquer les différences entre les RNN et les LSTM dans le cadre du POS Tagging. Nos recherches dans la littérature (cf Références) nous ont amenés à élargir notre étude en y incluant les GRU. Nous obtenons donc les résultats suivants:
  
   Architecture                            F1score sur le Test set
  
Vanilla RNN                                         72,16 %

RNN à deux couches	                              	73,31 %

RNN bidirectionnel	                              	78,11 %

LSTM simples	                              	      82,03 %

LTSM à deux couches	                              	85,02 %

LSTM bidirectionnel	                              	85,12 %

LSTM à deux couches bidirectionnel                 	85,88 %

GRU simple	                              	        75,36 %

GRU à deux couches	                              	80,92 %

GRU bidirectionnel	                              	75,59 %

  
  
 Ce git est organisé de la manière suivante :
 
-   datasetNLTK : Ce fichier permet le chargement du dictionnaire d'où sont issus les ensembles d'entraînement, de validation et de test. Nous utilisons le brown corpus fourni par la libraire NLTK de Python.
  
-   parameters : Ce fichier permet de définir les différents paramètres de l'étude ( la taille des ensembles, le nombre d'épisodes, le learning rate, etc...).
  
-   modelGRU : Ce fichier contient nos tros modèles de GRU, codés à partir de l'architecture Pytorch, le SimpleGRU, le BiGRU (GRU Bidirectionnel), le DoubleGRU(GRU à deux couches). Nous avons utilisé l'architecture fournie par la libraire PyTorch.
  
-   modelRNN : Ce fichier contient nos tros modèles de GRU, codés à partir de l'architecture Pytorch, le SimpleRNN, le BidirectionalRNN (RNN Bidirectionnel), le DoubleRNN(RNN à deux couches). Nous avons utilisé l'architecture fournie par la libraire PyTorch.
  
-   modelLSTM : Ce fichier contient nos tros modèles de GRU, codés à partir de l'architecture Pytorch, le LSTMTagger, le LSTMTagger_bidir (LSTM Bidirectionnel), le LSTMTagger_bidir_2layers(LSTM bidirectionnel à deux couches). Nous avons utilisé l'architecture fournie par la libraire PyTorch.
  
-   main : Ce fichier permet de lancer l'entraînement des réseaux neuronaux. Si vous souhaitez le lancer, pensez à créer un dossier "Images" là où vous sauvegarderez le programme Python afin que le programme puisse enregistrer correctement les images liées à chaque modèle ( Si cette étape n'est pas respectée, le programme risque de ne pas marcher correctement).
