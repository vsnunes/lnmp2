import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB

#processamento do ficheiro
training_data = []
corpus =  open('corpora/QuestoesConhecidas.txt', 'r')
with open('corpora/QuestoesConhecidas.txt') as corpus:
    for line in corpus:
        (firstWord, rest) = line.split(maxsplit = 1)
        training_data.append({"category": firstWord,"sentence": rest.rstrip()})


categories = list(set([a['category'] for a in training_data]))
class_words = []
corpus_words = []

    
for category in categories:
    words=[]
    for line in training_data:
        for word in nltk.word_tokenize(line['sentence']):
            if word not in ('?', ':', '.', ',' "'s"):
                corpus_words.append(word.lower())
                if line['category'] == category:
                    words.append(word.lower())
    class_words.append((words, category))

corpus_words = nltk.FreqDist(corpus_words)

#def create_training_set(class_words):

#MNB_classifier = SklearnClassifier(MultinomialNB())
#MNB_classifier.train(class_words)
#print("MultinomialNB accuracy percent:",nltk.classify.accuracy(MNB_classifier, class_words))
