from sklearn.datasets import fetch_20newsgroups
import helper as hlp
import nltk
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
import taskd as td
import taskd as tb

twenty_train, twenty_test = hlp.fetch_data()
hlp.classify_into_two_class(twenty_train)
hlp.classify_into_two_class(twenty_test)

tfidfListTrain = td.gettfidfListTrain()
nmfListTrain = td.getnmfListTrain()
tfidfListTest = td.gettfidfListTest()
nmfListTest = td.getnmfListTest()

classifier=MultinomialNB()

for min_df in [2,5]:
    print "WIth min_df = ", min_df
    if min_df == 2: 
        nmf_matrix_train=nmfListTrain[0]
        nmf_matrix_test=nmfListTest[0]
        tfidf_matrix_train=tfidfListTrain[0]
        tfidf_matrix_test=tfidfListTest[0]
    else: 
        nmf_matrix_train=nmfListTrain[1]
        nmf_matrix_test=nmfListTest[1]
        tfidf_matrix_train=tfidfListTrain[1]
        tfidf_matrix_test=tfidfListTest[1]

    print ".......... With SVD ........."
    classifier.fit(tfidf_matrix_train, twenty_train.target)
    predicted = classifier.predict(tfidf_matrix_test)
    probabilities = classifier.predict_proba(tfidf_matrix_test)
    hlp.getStats(twenty_test.target, predicted)
    hlp.plot_roc(twenty_test.target, probabilities[:,1], 'MultinomialNB')

    print ".......... With NMF .........."
    classifier.fit(nmf_matrix_train, twenty_train.target)
    predicted = classifier.predict(nmf_matrix_test)
    probabilities = classifier.predict_proba(nmf_matrix_test)
    hlp.getStats(twenty_test.target, predicted)
    hlp.plot_roc(twenty_test.target, probabilities[:,1], 'MultinomialNB')
    