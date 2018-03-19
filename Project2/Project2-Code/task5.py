import helper as hlp
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
import string
import nltk
import numpy as np
import re
from sklearn.datasets import fetch_20newsgroups
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize          
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import sklearn.metrics as smet
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.decomposition import PCA as sklearnPCA

print "..............Performing tfidf on full dataset............"
dataset = hlp.fetch_all_data()
count_vect = hlp.get_CountVectorizer(3)
count_transformed=count_vect.fit_transform(dataset.data)
transformer = hlp.get_TFIDFTransformer()
tfidf_matrix=transformer.fit_transform(count_transformed)
print "Dimensions of the TFIDF matrix is ", tfidf_matrix.shape

def svd_main():
    svd=TruncatedSVD(n_components=310, random_state=42)
    return svd

"""Task 3b for 20 classes"""
print "Best r for 20 classes"
rank_list = [1,2,3,5,10,20,50,100,300]

homo_list_svd = []
comp_list_svd = []
vscore_list_svd = []
adjscore_list_svd = []
infoscore_list_svd = []

homo_list_nmf = []
comp_list_nmf = []
vscore_list_nmf = []
adjscore_list_nmf = []
infoscore_list_nmf = []

svdtemp = svd_main()
svd_matrix = svdtemp.fit_transform(tfidf_matrix)
labels = dataset.target

def computeBestR():
    for r in rank_list:
        print "................. For r=",r,"......................\n"
        svd=svd_matrix[:,0:r]
        nmf = hlp.getNMF(r)
        nmf_matrix = nmf.fit_transform(tfidf_matrix)

        km = hlp.getKmeans(20)

        print "*******With LSI********"
        km.fit(svd)
        homo, comp, vscore, adjscore, infoscore = hlp.getStats(labels,km.labels_)
        homo_list_svd.append(homo)
        comp_list_svd.append(comp)
        vscore_list_svd.append(vscore)
        adjscore_list_svd.append(adjscore)
        infoscore_list_svd.append(infoscore)
        print ""
        
        
        print "*******With NMF********"
        km.fit(nmf_matrix)
        homo, comp, vscore, adjscore, infoscore = hlp.getStats(labels,km.labels_)
        homo_list_nmf.append(homo)
        comp_list_nmf.append(comp)
        vscore_list_nmf.append(vscore)
        adjscore_list_nmf.append(adjscore)
        infoscore_list_nmf.append(infoscore)

    print ".............With LSI............."
    plt.plot(rank_list, homo_list_svd)
    plt.ylabel('Homogeneity Score')
    plt.show()

    plt.plot(rank_list, comp_list_svd)
    plt.ylabel('Completeness Score')
    plt.show()

    plt.plot(rank_list, vscore_list_svd)
    plt.ylabel('V-measure Score')
    plt.show()

    plt.plot(rank_list, adjscore_list_svd)
    plt.ylabel('Adjusted rand Score')
    plt.show()

    plt.plot(rank_list, infoscore_list_svd)
    plt.ylabel('Adjusted Mutual Info Score')
    plt.show()
    
    print "............With NMF............."
    plt.plot(rank_list, homo_list_nmf)
    plt.ylabel('Homogeneity Score')
    plt.show()

    plt.plot(rank_list, comp_list_nmf)
    plt.ylabel('Completeness Score')
    plt.show()

    plt.plot(rank_list, vscore_list_nmf)
    plt.ylabel('V-measure Score')
    plt.show()

    plt.plot(rank_list, adjscore_list_nmf)
    plt.ylabel('Adjusted rand Score')
    plt.show()

    plt.plot(rank_list, infoscore_list_nmf)
    plt.ylabel('Adjusted Mutual Info Score')
    plt.show()

def computeWithScaling():
    print "-------------- Scaled SVD----------------"
    svd_old=svd_matrix[:,0:100]
    scaled_svd_matrix = preprocessing.scale(svd_old, with_mean = False)
    kmeans = hlp.getKmeans(20)
    svd_new=hlp.getSVD(2)
    svd_matrix_new = svd_new.fit_transform(scaled_svd_matrix)
    kmeans.fit(scaled_svd_matrix)
    hlp.plot20Clusters(svd_matrix_new,kmeans,"clusters_svd_scaled_20classes.png")
    hlp.getStats(labels,kmeans.labels_)
    
    print "--------------Scaled NMF----------------"
    nmf_old = hlp.getNMF(10)
    nmf_matrix = nmf_old.fit_transform(tfidf_matrix)
    scaled_nmf_matrix = preprocessing.scale(nmf_matrix, with_mean = False)
    kmeans = hlp.getKmeans(20)
    nmf_new=hlp.getNMF(2)
    nmf_matrix_new=nmf_new.fit_transform(scaled_nmf_matrix)
    kmeans.fit(scaled_nmf_matrix)
    hlp.plot20Clusters(nmf_matrix_new,kmeans,"clusters_nmf_scaled_20classes.png")
    hlp.getStats(labels,kmeans.labels_)
    
    print "--------------Logarithmic NMF----------------"
    nmf_matrix = nmf_old.fit_transform(tfidf_matrix)
    log_matrix = np.log(nmf_matrix+1)
    nmf_new=hlp.getNMF(2)
    nmf_matrix_new=nmf_new.fit_transform(log_matrix)
    kmeans.fit(log_matrix)
    hlp.plot20Clusters(nmf_matrix_new,kmeans,"clusters_nmf_log_20classes.png")
    hlp.getStats(labels,kmeans.labels_)
    
    print "--------------Log scaled NMF----------------"
    nmf_matrix = nmf_old.fit_transform(tfidf_matrix)
    log_matrix = np.log(nmf_matrix+1)
    nmf_matrix_scaled = preprocessing.scale(log_matrix, with_mean = False)
    nmf_new=hlp.getNMF(2)
    nmf_matrix_new=nmf_new.fit_transform(nmf_matrix_scaled)
    kmeans.fit(nmf_matrix_scaled)
    hlp.plot20Clusters(nmf_matrix_new,kmeans,"clusters_nmf_log_scaled_20classes.png")
    hlp.getStats(labels,kmeans.labels_)
    
    print "--------------Scaled log NMF----------------"
    nmf_matrix = nmf_old.fit_transform(tfidf_matrix)
    scaled_matrix = preprocessing.scale(nmf_matrix, with_mean = False)
    log_scaled_nmf = np.log(scaled_matrix+1)
    nmf_new=hlp.getNMF(2)
    nmf_matrix_new=nmf_new.fit_transform(log_scaled_nmf)
    kmeans.fit(log_scaled_nmf)
    hlp.plot20Clusters(nmf_matrix_new,kmeans,"clusters_nmf_scaled_log_20classes.png")
    hlp.getStats(labels,kmeans.labels_)

def compute4a():
    #convert HD to 2D
	print "........With LSI........"
	svd_old =svd_matrix[:,0:100]
	kmeans = hlp.getKmeans(20)
	svd_new=hlp.getSVD(2)
	svd_matrix_new = svd_new.fit_transform(svd_old)
	kmeans.fit(svd_old)
	hlp.plot20Clusters(svd_matrix_new,kmeans,"clusters_2d_svd_best_20classes.png")
	hlp.getStats(labels,kmeans.labels_)
	
	print ".........With NMF......."
	nmf = hlp.getNMF(10)
	nmf_matrix = nmf.fit_transform(tfidf_matrix)
	nmf_new=hlp.getNMF(2)
	nmf_matrix_new=nmf_new.fit_transform(nmf_matrix)
	kmeans.fit(nmf_matrix)
	hlp.plot20Clusters(nmf_matrix_new,kmeans,"clusters_2d_nmf_best_20classes.png")
	hlp.getStats(labels,kmeans.labels_)
    
computeBestR()
compute4a()
computeWithScaling()