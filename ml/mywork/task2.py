""" Groups and Topics

The objective of this task is to explore the structure of the deals.txt file. 

Building on task 1, we now want to start to understand the relationships to help us understand:

1. What groups exist within the deals?
2. What topics exist within the deals?

"""
import string
import os

#import logging
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import numpy

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

def create_dict_and_corpus(doc_fname,dict_fname,mm_corpus_fname):
    """ Returns dictionary and corpus by processing documents
    
    dictionary is saved to dict_fname for future access, 
    corpus is saved in MM format to file corpus_fname
    
    """
    dictionary=None
    corpus=None
    if(os.path.isfile(dict_fname)): #check with any file
        dictionary=corpora.Dictionary.load(dict_fname)
        corpus=corpora.MmCorpus(mm_corpus_fname)
    else:
        docs=process_docs(doc_fname) #preprocess documents by tokenizing
        dictionary=create_dict(docs)
        dictionary.save(dict_fname)
        corpus=create_corpus(docs,dictionary)
        corpora.MmCorpus.serialize(mm_corpus_fname,corpus)
    return dictionary,corpus

def model_with_LSI(corpus,dictionary,num_of_topics,lsi_file):
    """ Returns LSI topic model trained on corpus and dictionary with for number of topics 
    given by 'num_of_topics'. If the lsi_file already exist then instead of learning a model
    the stored model is loaded. 
    
    The model is saved to lsi_file when the model is trained.
    
    """
    lsi=None
    if(os.path.isfile(lsi_file)):
        #load model instead of training
        lsi=models.LsiModel.load(lsi_file)
    else:
        # Train LSI for topic modeling
        lsi=models.lsimodel.LsiModel(corpus=corpus, id2word=dictionary, num_topics=num_of_topics)
        lsi.save(lsi_file)
    return lsi


def model_with_LDA(corpus,dictionary,num_of_topics,lda_file):
    """ Returns LDA topic model trained on corpus and dictionary with for number of topics 
    given by 'num_of_topics'. If the lda_file already exist then instead of learning a model
    the stored model is loaded. 
    
    The model is saved to lda_file when the model is trained.
    
    """
    lda=None
    if(os.path.isfile(lda_file)):
        #load model instead of training
        lda=models.LdaModel.load(lda_file)
    else:
        # Train LDA model for topic modelling -- using default parameters for now
        lda=models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_of_topics)
        lda.save(lda_file)
    return lda

def model_with_hdplda(corpus,dictionary,hdplda_file):
    """ Returns LDA topic model trained on corpus and dictionary with from HDP model. 
    HDP model being non-parametric, the number of topics is inferred. If the hdplda_file already exist then instead of learning a model
    the stored model is loaded. 
    
    The model is saved to hdplda_file when the model is trained.
    
    """
    hdplda=None
    if(os.path.isfile(hdplda_file)):
        #load model instead of training
        hdplda=models.LdaModel.load(hdplda_file)
    else:
        # Train HDP model for topic modelling -- using default parameters for now
        hdp=models.hdpmodel.HdpModel(corpus=corpus, id2word=dictionary)
        alpha,beta=hdp.hdp_to_lda() #only gives alpha and beta for LDA
        hdplda=models.LdaModel(id2word=hdp.id2word,num_topics=len(alpha), 
                               alpha=alpha, eta=hdp.m_eta)
        hdplda.expElogbeta = numpy.array(beta, dtype=numpy.float32)
        hdplda.save(hdplda_file)
    return hdplda


def topic_based_clustering(model,dictionary,corpus):
    return None        






# Create of Load the dictionary and corpus from the documents file
doc_fname='../data/deals.txt'
dict_fname='../models/deals.dict'
mm_corpus_fname='../models/deals.mm'
dictionary,corpus=create_dict_and_corpus(doc_fname,dict_fname,mm_corpus_fname)

num_of_topics=100

lsi_file='../models/lsi_100topics.model'
lsi=model_with_LSI(corpus,dictionary,num_of_topics,lsi_file)

lda_file='../models/lda_100topics.model'
lda=model_with_LDA(corpus,dictionary,num_of_topics,lsi_file)

hdplda_file='../models/hdplda.model'
hdplda=model_with_hdplda(corpus,dictionary,hdplda_file)


#train TF-IDF model and create 
#tfidf=models.TfidfModel(corpus)
#corpus_tfidf=tfidf[corpus]


#ldamodel=models.LdaModel.load('../models/lda_100topics.model')
#corpus_ldamodel=ldamodel[corpus_tfidf]

#ldatop=open('../results/ldatopics.txt','w')
#for topic in ldamodel.show_topics(10):
#    ldatop.write(str(topic)+"\n")

i=0
aa=hdplda[corpus]
for a in aa:
    if i<10:
        print a
        i = i+1
    else:
        break

