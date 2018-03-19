from sklearn import svm
import taskb as tb
import taskd as td
from sklearn.svm import SVC
from sklearn.datasets import make_classification
import sklearn.metrics as smet
import matplotlib.pyplot as plt
import helper as hlp
import numpy as np

twenty_train, twenty_test = hlp.fetch_data()

hlp.classify_into_two_class(twenty_train)
hlp.classify_into_two_class(twenty_test)

def classifyWithSVC(valC):
    clf = svm.SVC(C=valC, probability=True, kernel='linear', random_state=42)

    svdListTrain = td.getsvdListTrain()
    nmfListTrain = td.getnmfListTrain()
    svdListTest = td.getsvdListTest()
    nmfListTest = td.getnmfListTest()

    for min_df in [2,5]:
        print ".......... With min_df = ", min_df , "..........."
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

        print "With SVD"
        clf.fit(svd_matrix_train, twenty_train.target)
        predicted = clf.predict(svd_matrix_test)
        probabilities = clf.predict_proba(svd_matrix_test)
        hlp.getStats(twenty_test.target, predicted)
        hlp.plot_roc(twenty_test.target, probabilities[:,1], 'SVM')

        print "With NMF"
        clf.fit(nmf_matrix_train, twenty_train.target)
        predicted = clf.predict(nmf_matrix_test)
        probabilitiesnmf = clf.predict_proba(nmf_matrix_test)
        hlp.getStats(twenty_test.target, predicted)
        hlp.plot_roc(twenty_test.target, probabilitiesnmf[:,1], 'SVM')
    
classifyWithSVC(1000)
classifyWithSVC(0.001)