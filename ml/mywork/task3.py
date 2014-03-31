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
import string

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import neighbors
from sklearn import cross_validation, metrics


def get_data_list(file):
    """ Return the list of lines"""
    return [line.lower() for line in open(file)]
    
gooddata='../data/good_deals.txt'
baddata='../data/bad_deals.txt'
testdata='../data/test_deals.txt'



def get_matrices(good_deals,bad_deals,test_deals):
    """ Return the training and testing matrices with labels """
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
    
    return train_mat.todense(),labels,test_mat.todense()

def get_train_test_data(gooddata,baddata,testdata):
    """ This function reads data files and computes tfidf matrices for training and test data.
    Returns training and testing matrices, labels, and list of training and testing deals
    """
    # Get list of deals
    good_deals=get_data_list(gooddata)
    bad_deals=get_data_list(baddata)
    test_deals=get_data_list(testdata)
    train_mat,labels,test_mat=get_matrices(good_deals,bad_deals,test_deals)
    
    return train_mat,labels,test_mat,good_deals,bad_deals,test_deals

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
        

def is_number(s):
    """ Helper to return if the string represents a number of not"""
    try:
        float(s)
        return True
    except ValueError:
        return False

def separate_deals(gooddata):
    """ Parse the gooddata file and returns coupon requiring deals and coupon not requiring deals"""
    good_deals=get_data_list(gooddata)
    deals_with_coupon = []
    deals_without_coupon = []
    coupon_words = ['coupon','code','free']
    off = ['off']
    for line in good_deals:
        withcoupon=False
        words=line.translate(None,string.punctuation).split()
        for i in range(len(words)):
            if words[i] in coupon_words:
                deals_with_coupon.append(line)
                withcoupon=True
            elif i < (len(words)-1) and is_number(words[i]) and words[i+1] in off:
                deals_with_coupon.append(line)
                withcoupon=True
        if withcoupon is False:
            deals_without_coupon.append(line)
    return deals_with_coupon, deals_without_coupon 


def run_knn(nrange,weights,train_mat,ofile):
    """ This function runs KNN algorithm for tuning model using number of neighbors fron nrange and 
    weight from weights"""
    # Tune the weight and the number of neighbors 
    goodknn=None
    maxacc=0
    maxf1=0
    n=0;
    for n_neighbors in nrange:
        for weight in weights:
            knn = neighbors.KNeighborsClassifier(n_neighbors, weights=weight)    
            knn,acc,f1=run_classifier(knn,train_mat,labels)
            if(acc>maxacc):
                if(f1>=maxf1):
                    n=n_neighbors
                    maxacc=acc
                    maxf1=f1
                    goodknn=knn
            elif acc==maxacc:
                if(f1>maxf1):
                    n=n_neighbors
                    maxacc=acc
                    maxf1=f1
                    goodknn=knn
            ofile.write(str(n_neighbors)+" "+str(weight)+" "+str(acc)+" "+str(f1)+"\n")
    return goodknn,n,maxacc,maxf1
   
#generate matrix for training along with deals data
train_mat,labels,test_mat,good_deals,bad_deals,test_deals=get_train_test_data(gooddata,baddata,testdata)

# Tune the KNN model to get better accuracy and f1 score
ofile1=open('../output/predictionKNNGoodBadDeals.txt','w')
ofile1.write("Experiment with KNN Model\n\n Running KNN for differnt values of number of neighbors and weight\n")
ofile1.write("numofneighbors   weight   accuracy     f1score\n")

knn,n,acc,f1=run_knn(range(3,10),['uniform','distance'],train_mat,ofile1)

ofile1.write("Bad Good Deals with KNN : neighbors = "+str(n)+' accuracy = '+str(acc)+' F1 = '+str(f1)+"\n\n")
#irrespective of weights, n_neighbors=5 has the better result for both accuracy and f1 score

ofile1.write("Prediction with test deals\n\n")
# Predict the actual test data
prediction=knn.predict(test_mat)
for i in range(len(prediction)):
    if(prediction[i]==0):
        ofile1.write("Good -->  "+test_deals[i]+"\n")
    else:
        ofile1.write("Bad -->  "+test_deals[i]+"\n")

ofile1.close()

# Use Predicted information fro bad and good deals to predict coupon requirement

with_copupon, wihtout_coupon = separate_deals(gooddata)

#select only predcted good deals that needs to be testing
test_coupon = []
for i in range(len(prediction)):
    if prediction[i]==0: #bad_deal need not be processed for querying coupon requirement
        test_coupon.append(test_deals[i])

train_mat,labels,test_mat=get_matrices(with_copupon, wihtout_coupon,test_coupon)        

ofile1=open('../output/predictionKNNCouponDeals.txt','w')
ofile1.write("Experiment with KNN Model\n\n Running KNN for differnt values of number of neighbors and weight\n")
ofile1.write("numofneighbors   weight   accuracy     f1score\n")

knn,n,acc,f1=run_knn(range(3,10),['uniform','distance'],train_mat,ofile1)

ofile1.write("Coupon No-Coupon Deals with KNN : neighbors = "+str(n)+' accuracy = '+str(acc)+' F1 = '+str(f1)+"\n\n")

ofile1.write("Prediction with test deals\n\n")
prediction=knn.predict(test_mat)
for i in range(len(prediction)):
    if(prediction[i]==0):
        ofile1.write("Coupon -->  "+test_deals[i]+"\n")
    else:
        ofile1.write("NoCoupon -->  "+test_deals[i]+"\n")
    

ofile1.close()










    