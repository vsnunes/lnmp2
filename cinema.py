import sys

categories_training_solutions = []
categories_list = []
questions_list  = []
with open(sys.argv[1]) as corpus:
        for line in corpus:
            (category, question) = line.split(maxsplit = 1)
            categories_training_solutions.append(category)
            questions_list.append(question.rstrip())

            