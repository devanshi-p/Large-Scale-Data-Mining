from sklearn.datasets import fetch_20newsgroups
import helper as hlp
import taskd as td
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import sklearn.metrics as smet
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

categories=hlp.fetch_categories()
twenty_train , twenty_test = hlp.fetch_data()
hlp.classify_into_two_class(twenty_train)
hlp.classify_into_two_class(twenty_test)

svdListTrain = td.getsvdListTrain()
nmfListTrain = td.getnmfListTrain()
svdListTest = td.getsvdListTest()
nmfListTest = td.getnmfListTest()
classifier = LogisticRegression(C=10000)

def classifyLR(train,test):
    classifier.fit(train,twenty_train.target)
    predicted = classifier.predict(test)
    predicted_probs = classifier.predict_proba(test)
    hlp.getStats(twenty_test.target, predicted)
    hlp.plot_roc(twenty_test.target, predicted_probs[:,1], 'Logistic Regression')

for min_df in [2,5]:
    if min_df == 2:
        svd_matrix_train=svdListTrain[0]
        nmf_matrix_train=nmfListTrain[0]
        svd_matrix_test=svdListTest[0]
        nmf_matrix_test=nmfListTest[0]
    else: 
        svd_matrix_train=svdListTrain[1]
        nmf_matrix_train=nmfListTrain[1]
        svd_matrix_test=svdListTest[1]
        nmf_matrix_test=nmfListTest[1]  
            
    print "**********With SVD**********"
    classifyLR(svd_matrix_train, svd_matrix_test)
    
    print "**********With NMF************"
    classifyLR(nmf_matrix_train, nmf_matrix_test)
    
