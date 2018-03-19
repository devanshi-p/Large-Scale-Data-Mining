import helper as hlp
import taskd as td
import numpy as np
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.datasets import make_classification
import sklearn.metrics as smet
import matplotlib.pyplot as plt
import numpy as np

params = list(range(-3, 3))
scores = []

twenty_train, twenty_test = hlp.fetch_data()
hlp.classify_into_two_class(twenty_train)
hlp.classify_into_two_class(twenty_test)

svdListTrain = td.getsvdListTrain()
nmfListTrain = td.getnmfListTrain()

svdListTest = td.getsvdListTest()
nmfListTest = td.getnmfListTest()

def callClassifier(train):
    for param in params:
        classifier = SVC(C = 10 ** (param))
        scores.append(
            np.mean(
                cross_validation.cross_val_score(
                    classifier,
                    train,
                    twenty_train.target,
                    cv = 5
                )
            )
        )

    best_val = params[scores.index(max(scores))]
    print "best val is ",best_val
    clf_5fold = SVC(C = 10 ** (best_val), probability=True, kernel='linear',random_state=42)
    return clf_5fold


for min_df in [2,5]:
    print "Calculating for min_df = ", min_df
    if min_df == 2:
        svd_matrix_train=svdListTrain[0]
        svd_matrix_test=svdListTest[0]
        nmf_matrix_train=nmfListTrain[0]
        nmf_matrix_test=nmfListTest[0]
    else:
        svd_matrix_train=svdListTrain[1]
        svd_matrix_test=svdListTest[1]
        nmf_matrix_train=nmfListTrain[1]
        nmf_matrix_test=nmfListTest[1]
    
    print "************With SVD***********"
    clf_5fold_svd = callClassifier(svd_matrix_train)
    clf_5fold_svd.fit(svd_matrix_train, twenty_train.target)
    predicted = clf_5fold_svd.predict(svd_matrix_test)
    probabilities = clf_5fold_svd.predict_proba(svd_matrix_test)
    hlp.getStats(twenty_test.target, predicted)
    hlp.plot_roc(twenty_test.target, probabilities[:,1], 'SVM with cross-validation')
    
    print "************With NMF***********"
    clf_5fold_nmf = callClassifier(nmf_matrix_train)
    clf_5fold_nmf.fit(nmf_matrix_train, twenty_train.target)
    predictednmf = clf_5fold_nmf.predict(nmf_matrix_test)
    probabilitiesnmf = clf_5fold_nmf.predict_proba(nmf_matrix_test)
    hlp.getStats(twenty_test.target, predictednmf)    
    hlp.plot_roc(twenty_test.target, probabilitiesnmf[:,1], 'SVM with cross-validation')