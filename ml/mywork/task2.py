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
from gensim import corpora, models, matutils
from sklearn.cluster import KMeans
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

def create_dict_and_corpus(doc_fname,dict_fname):
    """ Returns dictionary and corpus by processing documents
    
    dictionary is saved to dict_fname for future access, 
    corpus is saved in MM format to file corpus_fname
    
    """
    dictionary=None
    corpus=None
    docs=process_docs(doc_fname) #preprocess documents by tokenizing
    if(os.path.isfile(dict_fname)): #check with any file
        dictionary=corpora.Dictionary.load(dict_fname)
    else:
        dictionary=create_dict(docs)
        dictionary.save(dict_fname)
    corpus=create_corpus(docs,dictionary)
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
        lda=models.ldamodel.LdaModel(corpus=corpus, num_topics=num_of_topics, id2word=dictionary)
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
    return hdplda, len(alpha)


def topic_based_kmeans(corpus,num_of_topics,k):
    """ Computed K-means clustering and returns the model
    
    corpus - MMCorpus that is changed to dense matrix for use in Scikit-learn KMeans computation
    num_of_topics - number of features for new corpus
    k - number of clusters
    
    """
    # convert corpus to dense matrix with dimentiosn number of documents, number of topics
    ncorpus=matutils.corpus2dense(corpus,num_of_topics)
    # Initialize KMeans from Scikit-learn
    kmeans=KMeans(k,init='k-means++')
    # Fit the model
    kmeans.fit(ncorpus)
    return kmeans  
    
def print_topics(model,n,k):
    """ Print n number of topics with k terms from the model"""
    id=0
    for topic in model.show_topics(n,k,formatted=False):
        print id, "  --->  ",
        for term in topic:
            print term[1], #print only word
        print "\n"
        id += 1
   

def print_topics_with_groups(model,n,k,group_label,of):
    """ Print n cluster ID, topic ID, and k terms"""
    id=0
    of.write("GroupID  --  TopicID  --->                Terms\n")
    for topic in model.show_topics(n,k,formatted=False):
        of.write(str(group_label[id])+"    --    "+str(id)+ "    --->   ")
        for term in topic:
            of.write(" "+str(term[1])) #print only word
        of.write("\n")
        id += 1
    of.write("\n")

def run_with_LSI(doc_fname,ofile):
    """ Generate topics and groups and print the result"""
    # Create of Load the dictionary and corpus from the documents file
    dict_fname='../models/deals.dict'
    
    dictionary,corpus=create_dict_and_corpus(doc_fname,dict_fname)
    
    num_of_topics=50
    
    # Load LSI model to generate topics and perfrom grouping
    lsi_file='../models/lsi.model'
    lsi=model_with_LSI(corpus,dictionary,num_of_topics,lsi_file)
    # Perform LSI transforamtion for corpus
    corpus_lsi=lsi[corpus]
    #Perform K-Means clustering -- number of clusters needs to be tuned
    kmeans_lsi=topic_based_kmeans(corpus_lsi,num_of_topics,20)
    # Print output
    of=open(ofile,'w')
    of.write('Topic Modelling with LSI -- K-Means with reduced dimension defined by LSI topics\n\n')    
    print_topics_with_groups(lsi,num_of_topics,10,kmeans_lsi.labels_,of)
    of.close()

def run_with_LDA(doc_fname,ofile):  
    """ Generate topics and groups and print the result"""
    # Create of Load the dictionary and corpus from the documents file
    dict_fname='../models/deals.dict'
    dictionary,corpus=create_dict_and_corpus(doc_fname,dict_fname)
    
    num_of_topics=50
    # Now perform topic modelling and Clustering using LDA 
    lda_file='../models/lda.model'
    lda=model_with_LDA(corpus,dictionary,num_of_topics,lda_file)
    corpus_lda=lda[corpus]
    kmeans_lda=topic_based_kmeans(corpus_lda,num_of_topics,20)
    # Print output
    of=open(ofile,'w')
    of.write('Topic Modelling with LDA -- K-Means with reduced dimension defined by LDA topics\n\n')
    print_topics_with_groups(lda,num_of_topics,10,kmeans_lda.labels_,of)
    of.close()    

def run_with_HDPLDA(doc_fname,ofile):
    """ Generate topics and groups and print the result"""
    # Create of Load the dictionary and corpus from the documents file
    dict_fname='../models/deals.dict'
    dictionary,corpus=create_dict_and_corpus(doc_fname,dict_fname)
    
    #num_of_topics=50    
    # Perform modelliing with HDP + LDA
    hdplda_file='../models/hdplda.model'
    hdplda,num_of_topics=model_with_hdplda(corpus,dictionary,hdplda_file)
    corpus_hdplda=hdplda[corpus]
    kmeans_hdplda=topic_based_kmeans(corpus_hdplda,num_of_topics,20)
    # Print output
    of=open(ofile,'w')
    of.write('Topic Modelling with HDP+LDA -- K-Means with reduced dimension defined by HDP+LDA topics\n\n')
    print_topics_with_groups(hdplda,num_of_topics,10,kmeans_hdplda.labels_)
    of.close()
    




