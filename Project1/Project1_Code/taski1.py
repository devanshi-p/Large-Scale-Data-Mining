from sklearn.datasets import fetch_20newsgroups
import helper as hlp
import taskd as td
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import sklearn.metrics as smet
import matplotlib.pyplot as plt
import numpy as np

categories=hlp.fetch_categories()
twenty_train , twenty_test = hlp.fetch_data()
hlp.classify_into_two_class(twenty_train)
hlp.classify_into_two_class(twenty_test)

svdListTrain = td.getsvdListTrain()
nmfListTrain = td.getnmfListTrain()
svdListTest = td.getsvdListTest()
nmfListTest = td.getnmfListTest()
coeff_list = [0.001,0.01,0.1,10,100,1000]

def plotGraph(testError,type):
    plt.plot(testError)
    titleName = "Testing errors for "+ type + " regularized logistic regression against the regularied coefficients"
    plt.title(titleName)
    plt.xlabel("Regularization coefficient")
    plt.ylabel("Testing Error")
    plt.xticks(range(6), [coeff for coeff in coeff_list])
    plt.show()
    
def classifyLR_WithReg(train,test):
    l1_hyperPlaneParameters = []
    l2_hyperPlaneParameters = []

    l1_testErrors = []
    l2_testErrors = []
    
    l1_accu = []
    l2_accu = []
    
    l1_p = []
    l2_p = []
    
    l1_r = []
    l2_r = []

    for coeff in coeff_list:
        print "************** ", coeff, "****************"
        l1_classifier = LogisticRegression(penalty='l1', C=coeff, solver = 'liblinear')
        l2_classifier = LogisticRegression(penalty='l2', C=coeff, solver = 'liblinear')

        l1_classifier.fit(train,twenty_train.target)
        l2_classifier.fit(train,twenty_train.target)

        l1_predictions=l1_classifier.predict(test)
        l2_predictions=l2_classifier.predict(test)

        l1_predicted_probs = l1_classifier.predict_proba(test)
        l2_predicted_probs = l2_classifier.predict_proba(test)
        hlp.plot_roc(twenty_test.target, l1_predicted_probs[:,1], 'LR with L1 regularization')
        hlp.plot_roc(twenty_test.target, l2_predicted_probs[:,1], 'LR with L2 regularization')

        l1_testErrors.append(100 - smet.accuracy_score(twenty_test.target,l1_predictions) * 100)
        l2_testErrors.append(100 - smet.accuracy_score(twenty_test.target,l2_predictions) * 100)
        
        l1_accu.append(smet.accuracy_score(twenty_test.target,l1_predictions) * 100)
        l2_accu.append(smet.accuracy_score(twenty_test.target,l2_predictions) * 100)
        
        l1_p.append(smet.precision_score(twenty_test.target,l1_predictions, average='macro') * 100)
        l2_p.append(smet.precision_score(twenty_test.target,l2_predictions, average='macro') * 100)
        
        l1_r.append(smet.recall_score(twenty_test.target,l1_predictions, average='macro') * 100)
        l2_r.append(smet.recall_score(twenty_test.target,l2_predictions, average='macro') * 100)

        l1_hyperPlaneParameters.append(np.mean(l1_classifier.coef_))
        l2_hyperPlaneParameters.append(np.mean(l2_classifier.coef_))
    
    plotGraph(l1_testErrors,"l1")
    plotGraph(l2_testErrors,"l2")
    index=0

    for coeff in coeff_list:
        print "coeff=", coeff
        print "Test error l1=", l1_testErrors[index]
        print "Mean of coeff l1=", l1_hyperPlaneParameters[index]
        print "Accuracy l1=", l1_accu[index]
        print "Precision l1=", l1_p[index]
        print "Recall l1=", l1_r[index]
        
        print "coeff=", coeff
        print "Test error l2=", l2_testErrors[index]
        print "Mean of coeff l2=", l2_hyperPlaneParameters[index]
        print "Accuracy l2=", l2_accu[index]
        print "Precision l2=", l2_p[index]
        print "Recall l2=", l2_r[index]
        index+=1
 
for min_df in [2,5]:
    print "......... min_df = ", min_df, "........."
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
    classifyLR_WithReg(svd_matrix_train,svd_matrix_test)
    
    
    print "**********With NMF************"
    classifyLR_WithReg(nmf_matrix_train,nmf_matrix_test) 