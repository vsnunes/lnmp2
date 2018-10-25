import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB

#processamento do ficheiro
def processFile(fileName):
    training_data = []
    with open(fileName) as corpus:
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

    word_features = list(corpus_words.keys())

    def find_features(txt):
        words = set(txt)
        features = {}
        for w in word_features:
            features[w] = (w in words)

        return features

    featuresets = [(find_features(rev), category) for (rev, category) in class_words]

    return featuresets

def merge(tagFile, questionFile):
    tF = open(tagFile, "r")
    qF = open(questionFile, "r")
    mF = open("corpora/TESTING.txt", "w")
    for line in tF:
        mF.write(line[:-1] + "\t" + qF.readline())
    tF.close()
    qF.close()
    mF.close()


training_set = processFile("corpora/QuestoesConhecidas.txt")

classifier = nltk.NaiveBayesClassifier.train(training_set)

testing_set = processFile("corpora/TESTING.txt")

print("NaiveBayesClassifier (nltk) accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)

#def create_training_set(class_words):

#MNB_classifier = SklearnClassifier(MultinomialNB())
#MNB_classifier.train(class_words)
#print("MultinomialNB accuracy percent:",nltk.classify.accuracy(MNB_classifier, class_words))

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import ComplementNB, MultinomialNB,BernoulliNB

testing_blind = []
for el in testing_set:
    testing_blind.append(el[0])

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MultinomialNB accuracy percent:",nltk.classify.accuracy(MNB_classifier, testing_set)*100)
print("*** TAGS: ****")
for el in testing_blind:
    print(MNB_classifier.classify(el))

BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
print("BernoulliNB accuracy percent:",nltk.classify.accuracy(BNB_classifier, testing_set)*100)
print("*** TAGS: ****")
for el in testing_blind:
    print(BNB_classifier.classify(el))

CNB_classifier = SklearnClassifier(ComplementNB())
CNB_classifier.train(training_set)
print("ComplementNB accuracy percent:",nltk.classify.accuracy(CNB_classifier, testing_set)*100)
print("*** TAGS: ****")
for el in testing_blind:
    print(CNB_classifier.classify(el))