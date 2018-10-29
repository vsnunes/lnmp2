import nltk
import numpy
import random
from nltk.stem.lancaster import LancasterStemmer
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import ComplementNB, MultinomialNB,BernoulliNB
from nltk.corpus import stopwords

nltk.download('stopwords')

training_data = []
corpus_words = []
categories = []

def processTrainFile(fileName):
    """Given a fileName with category and questions, parse them for previous classifier training.

    Args:
        fileName: The file name where the phrases are stored.
        Example of file struct:
            category_name1 [TAB] question1
            category_name2 [TAB] question2
                        ...

    Returns:
        !!

    """

    with open(fileName) as corpus:
        for line in corpus:
            (firstWord, rest) = line.split(maxsplit = 1)
            training_data.append({"category": firstWord,"sentence": rest.rstrip()})


    categories = list(set([a['category'] for a in training_data]))
    class_words = []
    corpus_words_freq = []
        
    stemmer = LancasterStemmer()
    
    for category in categories:
        words=[]
        for line in training_data:
            for word in nltk.word_tokenize(line['sentence']):
                
                # if the word is one of the stopwords (generic phrase articulators) ignore it
                if (word not in ('?', ':', '.', ',' "'s")) and (word not in stopwords.words('english')):
                    corpus_words.append(stemmer.stem(word))
                    if line['category'] == category:
                        words.append(stemmer.stem(word))
        class_words.append((words, category))

    corpus_words_freq = nltk.FreqDist(corpus_words)

    word_features = list(corpus_words_freq.keys())

    fs = lexeme_metric_training(class_words, word_features)

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

def lexeme_metric(lexemes_list, corpus_list):
    """Assign weight to UNCATEGORIZED lexemes based on some metric.

    Args:
        lexiemes_list: A list of lexemes to assign a weight.
        corpus_list: A list of all lexemes seen on training phrases.

    Returns:
        A dictionary containing the lexemes and a corresponding weight.
        Example: {'lexieme1': val1, 'lexieme2': val2, ... }
    """

    words = set(lexemes_list)
    metrics = {}
    for w in corpus_list:
        metrics[w] = (w in words)
    #for word in words:
    #    features[word] = 1
    
    return metrics

def lexeme_metric_training(categorized_lexemes_list, corpus_list):
    """Assign weight to CATEGORIZED lexemes based on some metric.

    Note: This function is very similiar to lexeme_metric. The difference relies on knowing the category of the lexemes.

    Args:
        categorized_lexemes_list: A list of tuples containing the lexeme and correponding category.
        Example: [ ('lexeme1', 'category1') , ('lexeme2', 'category2'),  ... ]

        corpus_list: A list of all lexemes seen on training phrases.

    Returns:
        A list of dictionaries containing the lexemes and theirs corresponding weights.
        Example: [{'lexieme1': val1, 'lexieme2': val2, ... }, ... ]
    """
    return [(lexeme_metric(lexeme, corpus_list), category) for (lexeme, category) in categorized_lexemes_list]

def tokenizePhrase(question):
    """Given a question, stem the words into lexemes ignoring stopwords (of, the, etc...)

    Args:
        question: A string corresponding to the question to stem.

    Returns:
        A list of lexemes.
        Example question: "What is my name?"
        Example returned list: ['what', 'nam']
    """

    stemmer = LancasterStemmer()
    tokenizedWords=[]
    for word in nltk.word_tokenize(question):
        # if the word is one of the stopwords (generic question articulators) ignore it
        if (word not in ('?', ':', '.', ',' "'s")) and (word not in stopwords.words('english')):
            tokenizedWords.append(stemmer.stem(word))
    return tokenizedWords

def parseToPredictor(question):
    """Given a question prepare them to the classifier, which means, tokenize and assign weights to the lexemes.

    Args:
        question: A string corresponding to the question we want to predict the category.

    Returns:
        A dictionary containing the lexemes and a corresponding weight.
    """
    tokens = tokenizePhrase(question)
    return lexeme_metric(tokens, corpus_words)

def predictor(classifier, question):
    """Given a question related to some movie, predicts it's category.

    Args:
        classifier: An instance of sklearn.SklearnClassifier object
        question: A string corresponding to the question we want to predict the category.

    Returns:
        A string corresponding to the predicted category name.
        Example (question): "Which is the most relevant actor in some movie?"
        Example (return)  : actor_name
    """
    d_fs = parseToPredictor(question)
    
    return classifier.classify(d_fs)


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

    
