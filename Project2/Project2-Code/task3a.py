import helper as hlp
import task1 as t1
import scipy.sparse.linalg as sl
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD

def getVariance(tfidf_matrix):
    svd=TruncatedSVD(n_components=1000, random_state=42)
    svd.fit(tfidf_matrix)
    S=svd.explained_variance_ratio_
    Q = S.cumsum()
    plt.plot(range(1,1001), Q)
    plt.ylabel('Ratio of variance retained')
    plt.show()
    
dataset = hlp.fetch_data()
hlp.classify_into_two_class(dataset)
tfidf_matrix = t1.getTFIDF_matrix(dataset,3)
getVariance(tfidf_matrix)