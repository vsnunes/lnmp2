import nltk
from nltk.stem.lancaster import LancasterStemmer
nltk.download('punkt')

stemmer = LancasterStemmer()

#processamento do corpus
training_data = []
corpus =  open('corpora/QuestoesConhecidas.txt', 'r')
with open('corpora/QuestoesConhecidas.txt') as corpus:
    for line in corpus:
        (firstWord, rest) = line.split(maxsplit = 1)
        training_data.append({"class": firstWord,"sentence": rest.rstrip()})

corpus_words = {}
class_words = {}

classes = list(set([a['class'] for a in training_data]))

for c in classes:
    class_words[c] = []

for line in training_data:
    for word in nltk.word_tokenize(line['sentence']):
        stemmed_word = stemmer.stem(word.lower())
        if stemmed_word not in corpus_words:
            corpus_words[stemmed_word] = 1
        else:
            corpus_words[stemmed_word] += 1
        class_words[line['class']].extend([stemmed_word])

print(class_words)
print(corpus_words)
