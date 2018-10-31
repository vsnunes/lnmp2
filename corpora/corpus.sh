#!/bin/bash
shuf corpus.txt | tee new_corpus.txt
head -n 175 new_corpus.txt > trainingset.txt
tail -n 75 new_corpus.txt > testingset.txt
egrep -o  "^\w+" testingset.txt > testingsetcat.txt
sed 's/[^\t ]* *//' testingset.txt > testingsetquestions.txt
