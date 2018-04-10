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
    
        $: dollar
        $ -$ --$ A$ C$ HK$ M$ NZ$ S$ U.S.$ US$
    '': closing quotation mark
        ' ''
    (: opening parenthesis
        ( [ {
    ): closing parenthesis
        ) ] }
    ,: comma
        ,
    --: dash
        --
    .: sentence terminator
        . ! ?
    :: colon or ellipsis
        : ; ...
    CC: conjunction, coordinating
        & 'n and both but either et for less minus neither nor or plus so
        therefore times v. versus vs. whether yet
    CD: numeral, cardinal
        mid-1890 nine-thirty forty-two one-tenth ten million 0.5 one forty-
        seven 1987 twenty '79 zero two 78-degrees eighty-four IX '60s .025
        fifteen 271,124 dozen quintillion DM2,000 ...
    DT: determiner
        all an another any both del each either every half la many much nary
        neither no some such that the them these this those
    EX: existential there
        there
    FW: foreign word
        gemeinschaft hund ich jeux habeas Haementeria Herr K'ang-si vous
        lutihaw alai je jour objets salutaris fille quibusdam pas trop Monte
        terram fiche oui corporis ...
    IN: preposition or conjunction, subordinating
        astride among uppon whether out inside pro despite on by throughout
        below within for towards near behind atop around if like until below
        next into if beside ...
    JJ: adjective or numeral, ordinal
        third ill-mannered pre-war regrettable oiled calamitous first separable
        ectoplasmic battery-powered participatory fourth still-to-be-named
        multilingual multi-disciplinary ...
    JJR: adjective, comparative
        bleaker braver breezier briefer brighter brisker broader bumper busier
        calmer cheaper choosier cleaner clearer closer colder commoner costlier
        cozier creamier crunchier cuter ...
    JJS: adjective, superlative
        calmest cheapest choicest classiest cleanest clearest closest commonest
        corniest costliest crassest creepiest crudest cutest darkest deadliest
        dearest deepest densest dinkiest ...
    LS: list item marker
        A A. B B. C C. D E F First G H I J K One SP-44001 SP-44002 SP-44005
        SP-44007 Second Third Three Two * a b c d first five four one six three
        two
    MD: modal auxiliary
        can cannot could couldn't dare may might must need ought shall should
        shouldn't will would
    NN: noun, common, singular or mass
        common-carrier cabbage knuckle-duster Casino afghan shed thermostat
        investment slide humour falloff slick wind hyena override subhumanity
        machinist ...
    NNP: noun, proper, singular
        Motown Venneboerger Czestochwa Ranzer Conchita Trumplane Christos
        Oceanside Escobar Kreisler Sawyer Cougar Yvette Ervin ODI Darryl CTCA
        Shannon A.K.C. Meltex Liverpool ...
    NNPS: noun, proper, plural
        Americans Americas Amharas Amityvilles Amusements Anarcho-Syndicalists
        Andalusians Andes Andruses Angels Animals Anthony Antilles Antiques
        Apache Apaches Apocrypha ...
    NNS: noun, common, plural
        undergraduates scotches bric-a-brac products bodyguards facets coasts
        divestitures storehouses designs clubs fragrances averages
        subjectivists apprehensions muses factory-jobs ...
    PDT: pre-determiner
        all both half many quite such sure this
    POS: genitive marker
        ' 's
    PRP: pronoun, personal
        hers herself him himself hisself it itself me myself one oneself ours
        ourselves ownself self she thee theirs them themselves they thou thy us
    PRP$: pronoun, possessive
        her his mine my our ours their thy your
    RB: adverb
        occasionally unabatingly maddeningly adventurously professedly
        stirringly prominently technologically magisterially predominately
        swiftly fiscally pitilessly ...
    RBR: adverb, comparative
        further gloomier grander graver greater grimmer harder harsher
        healthier heavier higher however larger later leaner lengthier less-
        perfectly lesser lonelier longer louder lower more ...
    RBS: adverb, superlative
        best biggest bluntest earliest farthest first furthest hardest
        heartiest highest largest least less most nearest second tightest worst
    RP: particle
        aboard about across along apart around aside at away back before behind
        by crop down ever fast for forth from go high i.e. in into just later
        low more off on open out over per pie raising start teeth that through
        under unto up up-pp upon whole with you
    SYM: symbol
        % & ' '' ''. ) ). * + ,. < = > @ A[fj] U.S U.S.S.R * ** ***
    TO: "to" as preposition or infinitive marker
        to
    UH: interjection
        Goodbye Goody Gosh Wow Jeepers Jee-sus Hubba Hey Kee-reist Oops amen
        huh howdy uh dammit whammo shucks heck anyways whodunnit honey golly
        man baby diddle hush sonuvabitch ...
    VB: verb, base form
        ask assemble assess assign assume atone attention avoid bake balkanize
        bank begin behold believe bend benefit bevel beware bless boil bomb
        boost brace break bring broil brush build ...
    VBD: verb, past tense
        dipped pleaded swiped regummed soaked tidied convened halted registered
        cushioned exacted snubbed strode aimed adopted belied figgered
        speculated wore appreciated contemplated ...
    VBG: verb, present participle or gerund
        telegraphing stirring focusing angering judging stalling lactating
        hankerin' alleging veering capping approaching traveling besieging
        encrypting interrupting erasing wincing ...
    VBN: verb, past participle
        multihulled dilapidated aerosolized chaired languished panelized used
        experimented flourished imitated reunifed factored condensed sheared
        unsettled primed dubbed desired ...
    VBP: verb, present tense, not 3rd person singular
        predominate wrap resort sue twist spill cure lengthen brush terminate
        appear tend stray glisten obtain comprise detest tease attract
        emphasize mold postpone sever return wag ...
    VBZ: verb, present tense, 3rd person singular
        bases reconstructs marks mixes displeases seals carps weaves snatches
        slumps stretches authorizes smolders pictures emerges stockpiles
        seduces fizzes uses bolsters slaps speaks pleads ...
    WDT: WH-determiner
        that what whatever which whichever
    WP: WH-pronoun
        that what whatever whatsoever which who whom whosoever
    WP$: WH-pronoun, possessive
        whose
    WRB: Wh-adverb
        how however whence whenever where whereby whereever wherein whereof why
    ``: opening quotation mark
        ` ``

"""
#import nltk
#nltk.download('brown')
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

"""class RNN(nn.Module):
#    input_size – The number of expected features in the input x
#    hidden_size – The number of features in the hidden state h
#    num_layers – Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two RNNs together to form a stacked RNN, with the second RNN taking in outputs of the first RNN and computing the final results. Default: 1
#    nonlinearity – The non-linearity to use. Can be either ‘tanh’ or ‘relu’. Default: ‘tanh’
#    bias – If False, then the layer does not use bias weights b_ih and b_hh. Default: True
#    batch_first – If True, then the input and output tensors are provided as (batch, seq, feature)
#    dropout – If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer, with dropout probablity equal to dropout. Default: 0
#    bidirectional – If True, becomes a bidirectional RNN. Default: False

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
        return result """

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


#model1 = SimpleRNN(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
model = DoubleRNN(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
#model3 = MultiRNN(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix), 10)
#model4 = TestRNN(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
inputs = prepare_sequence(training_data[0][0], word_to_ix)
tag_scores = model(inputs)
print(tag_scores)

Loss_average = []
Loss = []
epochLoss = []
for epoch in range(50):
    print(epoch)
    epochLoss = []
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
        epochLoss.append(loss.data[0])
        Loss.append(loss)
    Loss_average.append(np.average(epochLoss))

t_fin_training = time()

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



## Affichage des fonctions de perte
Loss_array = np.array(Loss)
Loss_array = Loss_array.flatten()
plt.plot(Loss_average)
plt.figure()
plt.plot(Loss_array)

print("Temps de generation du dictionnaire (s) ", t_fin_dict - t_init)
print("Temps d'entrainement du modele choisi (s) ", t_fin_training - t_fin_dict)
