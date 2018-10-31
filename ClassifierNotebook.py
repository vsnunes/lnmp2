#!/usr/bin/env python
# coding: utf-8

# In[49]:
import nltk

#http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

categories_training_solutions = []
categories_list = []
questions_list  = []
with open("corpora/QuestoesConhecidas.txt") as corpus:
        for line in corpus:
            (category, question) = line.split(maxsplit = 1)
            categories_training_solutions.append(category)
            questions_list.append(question.rstrip())


# In[3]:



# In[4]:


categories_list = list(set(categories_training_solutions))
categories_list


# In[5]:


#Text preprocessing, tokenizing and filtering of stopwords are all included in CountVectorizer, which builds a dictionary of features and transforms documents to feature vectors:

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_question_counts = count_vect.fit_transform(questions_list)
X_question_counts.shape


# In[6]:


#Term Frequency times Inverse Document Frequency

from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_question_counts)
X_question_tf = tf_transformer.transform(X_question_counts)
X_question_tf.shape


# In[7]:


#Prepare data to fit the estimator
tfidf_transformer = TfidfTransformer()
X_question_tfidf = tfidf_transformer.fit_transform(X_question_counts)
X_question_tfidf.shape


# In[8]:


### TRAINING CLASSIFIER ###
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_question_tfidf, categories_training_solutions)


# In[9]:


#Predict test
docs_new = ['What budget do I own?']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)
predicted

# In[11]:


from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
])


# In[12]:


text_clf.fit(questions_list, categories_training_solutions)


# In[13]:


import numpy as np
docs_test = questions_list
predicted = text_clf.predict(docs_test)
np.mean(predicted == categories_training_solutions) 


# In[14]:


#Lets load the testing sets
testing_questions_list = []
with open("corpora/NovasQuestoes.txt") as testing_corpus:
        for line in testing_corpus:
            question = line.rstrip()
            testing_questions_list.append(question)

testing_solutions_list = []
with open("corpora/NovasQuestoesResultados.txt") as testing_solutions_corpus:
        for line in testing_solutions_corpus:
            category = line.rstrip()
            testing_solutions_list.append(category)


# In[15]:


#Let’s see if we can do better with a linear support vector machine (SVM), which is widely regarded as one of 
#the best text classification algorithms (although it’s also a bit slower than naïve Bayes).
#We can change the learner by simply plugging a different classifier object into our pipeline:
#original hinge

from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, random_state=42,
                                           max_iter=5, tol=None)),
])
text_clf.fit(questions_list, categories_training_solutions)  

predicted = text_clf.predict(testing_questions_list)
np.mean(predicted == testing_solutions_list)


# In[45]:


text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='epsilon_insensitive', penalty='l2',
                                           alpha=1e-3, random_state=42,
                                           max_iter=5, tol=None)),
])
text_clf.fit(questions_list, categories_training_solutions)  

predicted = text_clf.predict(testing_questions_list)
np.mean(predicted == testing_solutions_list)


# In[58]:


from sklearn.feature_extraction.text import HashingVectorizer
text_clf = Pipeline([('vect', HashingVectorizer(analyzer='word')),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='epsilon_insensitive', penalty='l2',
                                           alpha=1e-3, random_state=42,
                                           max_iter=5, tol=None)),
])
text_clf.fit(questions_list, categories_training_solutions)  

predicted = text_clf.predict(testing_questions_list)
np.mean(predicted == testing_solutions_list)

# In[19]:

from sklearn.linear_model import SGDRegressor
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
classifiers.append(("EI HEHEHE", SGDClassifier(loss='epsilon_insensitive', penalty='l2',
                                           alpha=1e-3, random_state=42,
                                           max_iter=10, tol=None, learning_rate="invscaling", eta0=6)))
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


# In[ ]:




