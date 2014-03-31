""" Classification

The objective of this task is to build a classifier that can tell us whether a new, unseen deal 
requires a coupon code or not. 

We would like to see a couple of steps:

1. You should use bad_deals.txt and good_deals.txt as training data
2. You should use test_deals.txt to test your code
3. Each time you tune your code, please commit that change so we can see your tuning over time

Also, provide comments on:
    - How general is your classifier?
    - How did you test your classifier?

"""
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import neighbors
from sklearn import cross_validation, metrics


def get_data_list(file):
    """ Return the list of lines"""
    return [line for line in open(file)]
    
gooddata='../data/good_deals.txt'
baddata='../data/bad_deals.txt'
testdata='../data/test_deals.txt'

def get_train_test_data(gooddata,baddata,testdata):
    """ This function reads data files and computes tfidf matrices for training and test data.
    Returns training and testing matrices, labels, and list of training and testing deals
    """
    # Get list of deals
    good_deals=get_data_list(gooddata)
    bad_deals=get_data_list(baddata)
    test_deals=get_data_list(testdata)
    # Generte labels for good and bad deals
    labels=[0]*len(good_deals)+[1]*len(bad_deals)
    
    deals = good_deals+bad_deals
    
    # Instance of vectorizer that records counts of terms
    count_vectorizer = CountVectorizer()
    
    # Fit training and testing data to transfroms into matrix
    train=count_vectorizer.fit_transform(deals)    
    test = count_vectorizer.transform(test_deals)
    
    # Initialize TFIDF tranformer and transform testing and training data to tfidf matrix
    tfidf = TfidfTransformer()
    tfidf.fit(test)
    tfidf.fit(train)
    train_mat=tfidf.transform(train)
    test_mat = tfidf.transform(test)
    
    return train_mat.todense(),labels,test_mat.todense(),good_deals,bad_deals,test_deals


def run_classifier(classifier,train_mat,labels):
    """ This is a generic classifier that splits training data to training set and holdout set 
    using cross-validation. Returns fitted model, accuracy and  f1 score performance measures.
    
    classifier -- classifier function
    train_mat -- training data matrix
    labels -- labels for training data
    
    """
    # Split the training data to generate trainining set and hold-out set
    train_data, holdout_data, train_label, holdout_label = \
                    cross_validation.train_test_split(train_mat, labels, test_size=0.25, random_state=1979)
    # Learn the model
    classifier.fit(train_data,train_label)
    # Predict the test set
    prediction=classifier.predict(holdout_data)
    #compute f1score
    f1score=metrics.f1_score(holdout_label,prediction)
    accuracy=metrics.accuracy_score(holdout_label,prediction)
    
    return classifier,accuracy,f1score                
        
    
#generate matrix for training along with deals data
train_mat,labels,test_mat,good_deals,bad_deals,test_deals=get_train_test_data(gooddata,baddata,testdata)



n_neighbors=7
weights='uniform'

knn = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)

knn,acc,f1=run_classifier(knn,train_mat,labels)
print acc, f1
prediction=knn.predict(test_mat)



print len(prediction)










    