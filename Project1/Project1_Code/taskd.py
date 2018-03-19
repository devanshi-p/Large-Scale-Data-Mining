import helper as hlp
import taskb as tb
from sklearn.decomposition import NMF

twenty_train, twenty_test = hlp.fetch_data()

svdListTrain = []
svdListTest = []
nmfListTrain = []
nmfListTest = []
tfidfListTrain=[]
tfidfListTest=[]
for min_df in [2,5]:
    
    ############## For SVD ################
    tfidf_train_matrix, tfidf_test_matrix=tb.getTFIDF_matrix(twenty_train, twenty_test,min_df)
    
    tfidfListTrain.append(tfidf_train_matrix)
    tfidfListTest.append(tfidf_test_matrix)
    
    svd = hlp.getSVD()
    svd_matrix_train=svd.fit_transform(tfidf_train_matrix)
    svd_matrix_test=svd.transform(tfidf_test_matrix)
    
    svdListTrain.append(svd_matrix_train)
    svdListTest.append(svd_matrix_test)
    
    print "Shape of svd matrix for min_df = ", min_df, " is ", svd.components_.shape

    ############### For NMF #################
    nmfModel = hlp.getNMF()
    W_train = nmfModel.fit_transform(tfidf_train_matrix)
    W_test = nmfModel.transform(tfidf_test_matrix)
    
    nmfListTrain.append(W_train)
    nmfListTest.append(W_test)
    
    nmf = nmfModel.components_
    print "Shape of nmf matrix for min_df = ", min_df, " is ", nmf.shape

def getsvdListTrain():
    return svdListTrain

def getnmfListTrain():
    return nmfListTrain

def getsvdListTest():
    return svdListTest

def getnmfListTest():
    return nmfListTest

def gettfidfListTrain():
    return tfidfListTrain

def gettfidfListTest():
    return tfidfListTest