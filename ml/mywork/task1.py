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

import nltk
from nltk.corpus import stopwords


counter=Counter()

def create_counter(corpus_fname,punctuation=string.punctuation,stop_words=stopwords.words("english")):
    """ This function reads the text file 'corpus_fname', tokenizes the text file, and
    creates a list of words->count 
    
    corpus_fname - data file name to be explored 
    punctuation - set off punctuation symbols to be removed; 
    stop_words - list of stop words; default is stopwords from nltk data
    
    """
    raw_corpus=open(corpus_fname)
    data_list=[]
    #create a tokenized list of deals data, where all punctuations and english stopwords are removed
    data_list=[[word for word in line.lower().translate(None,punctuation).split() 
                if word not in stop_words] for line in raw_corpus] 

    #count the occurrences of the word in modified data
    for vec in data_list:
        for word in vec:
            counter[word] += 1
    return counter        
    
    
def most_popular_term(n):
    """ Return n most frequent terms from the modified corpus"""
    return counter.most_common(n)

def least_popular_term(n):
    """ Return n least frequent terms from the modified corpus"""
    return counter.most_common()[-n:]

def guitar_types(raw_corpus):
    """ Return the unique list of guitar types

    This function collects the list of tokenzed sentences having the word 'guitar' from 'raw_corpus' 
    to identify guitar types.    
    
    The problem with this approach is if a word 'guitar' is qualified with an adjective that doesnot 
    specify type, we generate wrong result
        for example - decalgirl.com guitar is treated as a type
    """
    #tokenize the sentences from the corpus    
    sentences= nltk.sent_tokenize(raw_corpus.read().lower())  
    #tokenize words from each sentence in a corpus
    tokenized_sent=[nltk.word_tokenize(sent) for sent in sentences]
    #select only those sentences having the owrd 'guitar' in it and do pos tagging
    sentence_list=[nltk.pos_tag(tsent) for tsent in tokenized_sent if 'guitar' in tsent]
    #grammar rule for finding guitar type -- adjacective followed by noun
    rule=r"""ADP: {<JJ><NN>}"""
    chunker=nltk.RegexpParser(rule)
    types=set([])
    for sent in sentence_list:
        #generate parse tree for each sentence using grammar
        chunks=chunker.parse(sent)
        for stree in chunks.subtrees():
            if stree.node == 'ADP': 
                # In this program leaves is a list of two tagged words, 
                #where tagged word is a tuple (word,pos)
                leaves=stree.leaves()           
                if 'guitar' in leaves[1]: types.add(leaves[0][0]) 
    return list(types)  
    
def run(fname):
    create_counter(fname)
    print most_popular_term(1)
    print least_popular_term(1)
    print len(guitar_types)

run('../data/deals.txt')
    