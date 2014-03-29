""" Features

The objective of this task is to explore the corpus, deals.txt. 

The deals.txt file is a collection of deal descriptions, separated by a new line, from which 
we want to glean the following insights:

1. What is the most popular term across all the deals?
2. What is the least popular term across all the deals?
3. How many types of guitars are mentioned across all the deals?

"""
import string
from collections import Counter

#import nltk
from nltk.corpus import stopwords

#nltk.download("stopwords")
counter=Counter()
def explore_corpus(corpus_fname):
    """ This function reads the text file 'corpus_fname', tokenizes the text file, and
    creates a list of words->count to produce the most popular and the least popular terms 
    
    corpus_fname - data file name to be explored 
    
    """
    raw_corpus=open(corpus_fname)
    sentence_list=[[word for word in line.lower().translate(None,string.punctuation).split() 
                    if word not in stopwords.words("english")] for line in raw_corpus] 

    for vec in sentence_list:
        for word in vec:
            counter[word] += 1
    print most_popular_term(1)
    print least_popular_term(1)
    #print guitar_types_count(sentence_list)
    
def most_popular_term(n):
    return counter.most_common(n)

def least_popular_term(n):
    return counter.most_common()[-n:]
    
explore_corpus('../data/deals.txt')