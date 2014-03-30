""" Groups and Topics

The objective of this task is to explore the structure of the deals.txt file. 

Building on task 1, we now want to start to understand the relationships to help us understand:

1. What groups exist within the deals?
2. What topics exist within the deals?

"""
import string

#import logging
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models

#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def process_docs(corpus_fname):
    """ Process the text file containing documents represented by each line
    
    This function returns the list of tokenized documents"""
    raw_corpus=open(corpus_fname)
    docs=[]
    for line in raw_corpus:
        docs.append(tokenize(line))
    return docs

def create_dict(docs):
    """ Return the dictionary object from the list of tokenized documents"""
    return corpora.Dictionary(docs)

def create_corpus(docs,dictionary):
    """ Create a corpus (Sparse Vectors) using the list of tokenized documents and dictionary"""
    return [dictionary.doc2bow(doc) for doc in docs]

def tokenize(doc):
    """ Tokenize each document by removing punctuations, stopwords, and applying Porter stemmer"""
    raw=doc.lower().translate(None,string.punctuation)
    stemmer=PorterStemmer()    
    tokens=[stemmer.stem(word) for word in nltk.word_tokenize(raw) if word not in stopwords.words("english")] 
    return tokens

# Commentout next 5 statements after first run as the results are saved into file for future use
#docs=process_docs('../data/deals.txt')    
#dictionary=create_dict(docs)
#dictionary.save('../models/deals.dict')
#corpus=create_corpus(docs,dictionary)
#corpora.MmCorpus.serialize('../models/deals.mm',corpus)

# Uncomment next 2 statements after first run
#dictionary=corpora.Dictionary.load('../models/deals.dict')
#corpus=corpora.MmCorpus('../models/deals.mm')

#train TF-IDF model and create 
#tfidf=models.TfidfModel(corpus)
#corpus_tfidf=tfidf[corpus]

# Train LDA model for topic modelling 
#ldamodel=models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=100)
#ldamodel.save('../models/lda_100topics.model')

ldamodel=models.LdaModel.load('../models/lda_100topics.model')
#corpus_ldamodel=ldamodel[corpus_tfidf]

ldatop=open('../results/ldatopics.txt','w')
for topic in ldamodel.show_topics(10):
    ldatop.write(str(topic)+"\n")

# Train LSI for topic modeling
#lsi=models.lsimodel.LsiModel(corpus=corpus, id2word=dictionary, num_topics=100)
#lsi.save('../models/lsi_100topics.model')

lsi=models.LsiModel.load('../models/lsi_100topics.model')
lsitop=open('../results/lsitopics.txt','w')
for topic in lsi.show_topics(10):
    lsitop.write(str(topic)+"\n")
