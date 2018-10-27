import nltk
import numpy
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import ComplementNB, MultinomialNB,BernoulliNB

training_data = []
corpus_words = []
categories = []

#processamento do ficheiro de treino
def processTrainFile(fileName):

    with open(fileName) as corpus:
        for line in corpus:
            (firstWord, rest) = line.split(maxsplit = 1)
            training_data.append({"category": firstWord,"sentence": rest.rstrip()})


    categories = list(set([a['category'] for a in training_data]))
    class_words = []
    corpus_words_freq = []
        
    for category in categories:
        words=[]
        for line in training_data:
            for word in nltk.word_tokenize(line['sentence']):
                if word not in ('?', ':', '.', ',' "'s"):
                    corpus_words.append(word.lower())
                    if line['category'] == category:
                        words.append(word.lower())
        class_words.append((words, category))

    corpus_words_freq = nltk.FreqDist(corpus_words)

    word_features = list(corpus_words_freq.keys())

    fs = featuresets(class_words, word_features)

    return fs

# Do not use: Not giving correct values!!
def NME(output, input):
    n = len(output)
    m = len(input)

    distance = numpy.zeros((n,m))

    distance[0,0] = 0

    for j in range(1,n):
        distance[j,0] = distance[j-1,0] + 1
    
    for i in range(1,m):
        distance[0,i] = distance[0, i-1] + 1
    
    for j in range(1,n):
        for i in range(1,m):
            if output[j - 1] == input[i - 1]:
                subs_cost = 0
            else:
                subs_cost = distance[j-1,i-1]
            distance[j,i] = min(distance[j-1,i] + 1, subs_cost, distance[j,i-1] + 1)

    print(distance)
    return distance[n - 1, m - 1]

def find_features(txt, word_features):
    words = set(txt)
    features = {}
    #for w in word_features:
        #features[w] = (w in words)
    for word in words:
        features[word] = 1
    
    return features

def featuresets(class_words, word_features):
    return [(find_features(rev, word_features), category) for (rev, category) in class_words]

# Tokenizes a phrase
def tokenizePhrase(phrase):
    words=[]
    for word in nltk.word_tokenize(phrase):
        if word not in ('?', ':', '.', ',' "'s"):
            words.append(word.lower())
    return words

#Auxiliar method just to return the phrase to specific
#tokenized dictionary
def parseToPredictor(phrase):
    tokens = tokenizePhrase(phrase)
    tuple_list = []

    for token in tokens:
        tuple_list.append(token)

    return [find_features(word, corpus_words) for word in tuple_list]

def predictor(classifier, phrase):
    fs = parseToPredictor(phrase)
    
    stats = {}
    for word in fs:
        tag = classifier.classify(word)
        if tag not in stats.keys():
            stats[tag] = 1
        else:
            stats[tag] += 1

    return stats


def merge(tagFile, questionFile):
    tF = open(tagFile, "r")
    qF = open(questionFile, "r")
    mF = open("corpora/TESTING.txt", "w")
    for line in tF:
        mF.write(line[:-1] + "\t" + qF.readline())
    tF.close()
    qF.close()
    mF.close()


training_set = processTrainFile("corpora/QuestoesConhecidas.txt")

classifier = nltk.NaiveBayesClassifier.train(training_set)

###### TRAINING SETS ######
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
#print("MultinomialNB accuracy percent:",nltk.classify.accuracy(MNB_classifier, testing_set)*100)

BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
#print("BernoulliNB accuracy percent:",nltk.classify.accuracy(BNB_classifier, testing_set)*100)

CNB_classifier = SklearnClassifier(ComplementNB())
CNB_classifier.train(training_set)
#print("ComplementNB accuracy percent:",nltk.classify.accuracy(CNB_classifier, testing_set)*100)


testing_set = []
#Opening testing file
with open("corpora/TESTING.txt") as testing:
        for line in testing:
            print(predictor(MNB_classifier, line))

    
