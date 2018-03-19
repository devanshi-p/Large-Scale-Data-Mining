from sklearn import svm
from sklearn.datasets import fetch_20newsgroups
import helper as hlp
import nltk
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF

categories = ['comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','misc.forsale','soc.religion.christian']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)


def classify(train,test,obj):
    classifier_ovo=OneVsOneClassifier(obj)
    classifier_ovr=OneVsRestClassifier(obj)
    
    classifier_ovo.fit(train,twenty_train.target)
    classifier_ovr.fit(train,twenty_train.target)
    
    print "Testing"
    predicted_ovo=classifier_ovo.predict(test)
    predicted_ovr=classifier_ovr.predict(test)
  
    print "One vs one"
    hlp.getStats(twenty_test.target,predicted_ovo)
    
    print "One vs Rest"
    hlp.getStats(twenty_test.target,predicted_ovr)
    
for min_df in [2,5]:
    print "Calculating for min_df = ", min_df
    count_vect = hlp.get_CountVectorizer(min_df)
    tfidf_transformer = TfidfTransformer(sublinear_tf=True)
    mysvd=TruncatedSVD(n_components=50)
    mynmf=hlp.getNMF()
    
    #for train data vectorizer-> tfidf transformer -> svd/nmf
    X_train_counts = count_vect.fit_transform(twenty_train.data) 
    X_train_tfidf=tfidf_transformer.fit_transform(X_train_counts)
    X_train_svd=mysvd.fit_transform(X_train_tfidf)
    X_train_nmf = mynmf.fit_transform(X_train_tfidf)

    #for test data vectorizer-> tfidf transformer -> svd/nmf
    X_test_counts=count_vect.transform(twenty_test.data)
    X_test_tfidf=tfidf_transformer.transform(X_test_counts)
    X_test_svd=mysvd.transform(X_test_tfidf)
    X_test_nmf = mynmf.transform(X_test_tfidf)
    
    print "################# For SVM ##################"
    print "****** For SVD *******"
    classify(X_train_svd,X_test_svd,svm.SVC(kernel='linear'))
    
    print "****** For NMF *******"
    classify(X_train_nmf,X_test_nmf,svm.SVC(kernel='linear'))
   
    print "################# For Naive Bayes ##################"
    print "****** For SVD *******"
    classify(X_train_svd,X_test_svd,GaussianNB())
    
    print "****** For NMF *******"
    classify(X_train_nmf,X_test_nmf,GaussianNB())
    
    print "---------------Done for min_df = ", min_df ,"--------------"