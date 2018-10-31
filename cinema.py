import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
import numpy as np


categories_training_solutions = []
categories_list = []
questions_list  = []
testing_questions_list = []
testing_solutions_list = []

# Debug function: only to show accuracy
def testStats(trainingQuestionsAndSolutions, testingQuestions, testingSolutions):
    openFiles(trainingQuestionsAndSolutions, testingQuestions, testingSolutions)
    #CLASSIFIERS TEST for SGD
    classifiers = []
    classifiers.append(("EI hinge", SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, random_state=42,
                                            max_iter=1000, tol=None)))
    classifiers.append(("EI log", SGDClassifier(loss='log', penalty='l2',
                                            alpha=1e-3, random_state=42,
                                            max_iter=10, tol=None)))
    classifiers.append(("EI modified_huber", SGDClassifier(loss='modified_huber', penalty='l2',
                                            alpha=1e-3, random_state=42,
                                            max_iter=10, tol=None)))
    classifiers.append(("EI squared_hinge", SGDClassifier(loss='squared_hinge', penalty='l2',
                                            alpha=1e-3, random_state=42,
                                            max_iter=10, tol=None)))
    classifiers.append(("EI perceptron", SGDClassifier(loss='perceptron', penalty='l2',
                                            alpha=1e-3, random_state=42,
                                            max_iter=10, tol=None)))
    classifiers.append(("EI l2", SGDClassifier(loss='epsilon_insensitive', penalty='l2',
                                            alpha=1e-3, random_state=42,
                                            max_iter=5, tol=None)))
    classifiers.append(("EI l1", SGDClassifier(loss='epsilon_insensitive', penalty='l1',
                                            alpha=1e-3, random_state=42,
                                            max_iter=5, tol=None)))
    classifiers.append(("EI learningrate", SGDClassifier(loss='epsilon_insensitive', penalty='l2',
                                            alpha=1e-3, random_state=42,
                                            max_iter=5, tol=None, learning_rate="invscaling", eta0=6)))
    classifiers.append(("EI constant", SGDClassifier(loss='epsilon_insensitive', penalty='l2',
                                            alpha=1e-3, random_state=42,
                                            max_iter=5, tol=None, learning_rate="constant", eta0=2)))


    for (name, classifier) in classifiers:
        text_clf2 = Pipeline([('vect', CountVectorizer(analyzer='word')),
                            ('tfidf', TfidfTransformer()),
                            ('clf', classifier),])
        text_clf2.fit(questions_list, categories_training_solutions)  

        predicted = text_clf2.predict(testing_questions_list)
        print("{}: {}".format(name, np.mean(predicted == testing_solutions_list)))

# Open Training and Testing files
def openFiles(trainingQuestionsAndSolutions, testingQuestions, testingSolutions=None):
    with open(trainingQuestionsAndSolutions) as corpus:
            for line in corpus:
                (category, question) = line.split(maxsplit = 1)
                categories_training_solutions.append(category)
                questions_list.append(question.rstrip())

    #Lets load the testing sets
    with open(testingQuestions) as testing_corpus:
            for line in testing_corpus:
                question = line.rstrip()
                testing_questions_list.append(question)

    if (testingSolutions != None):
        with open(testingSolutions) as testing_solutions_corpus:
                for line in testing_solutions_corpus:
                    category = line.rstrip()
                    testing_solutions_list.append(category)


## MAIN ##


if (len(sys.argv) > 2):
    if (len(sys.argv) > 3): #just for debug, if testingSolutions is provided then compute accuracy
        testStats(sys.argv[1], sys.argv[2], sys.argv[3])
        exit(0)
    openFiles(sys.argv[1], sys.argv[2])
else:
    print("Missing arguments: ")
    print("Usage: python3 cinema.py trainingQuestions testingQuestions [testingSolutions]")
    exit(1)

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, random_state=42,
                                           max_iter=1000, tol=None)),
])

text_clf.fit(questions_list, categories_training_solutions)  

predicted = text_clf.predict(testing_questions_list)

#print the tag for each sentence in testing set
for tag_predicted in predicted:
    print(tag_predicted)