import helper as hlp
import task1 as t1
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing


dataset = hlp.fetch_data()
labels = hlp.fetch_labels(dataset)
hlp.classify_into_two_class(dataset)

tfidf_matrix = t1.getTFIDF_matrix(dataset,3)
kmeans = hlp.getKmeans(2)

print "-------------- Scaled SVD----------------"
svd = hlp.getSVD(3)
svd_matrix = svd.fit_transform(tfidf_matrix)
scaled_svd_matrix = preprocessing.scale(svd_matrix, with_mean = False)
kmeans.fit(scaled_svd_matrix)
hlp.plotClusters(scaled_svd_matrix,kmeans,"clusters_svd_scaled.png")
hlp.getStats(labels,kmeans.labels_)

print "--------------Scaled NMF----------------"
nmf = hlp.getNMF(10)
nmf_matrix = nmf.fit_transform(tfidf_matrix)
scaled_nmf_matrix = preprocessing.scale(nmf_matrix, with_mean = False)
kmeans.fit(scaled_nmf_matrix)
hlp.plotClusters(scaled_nmf_matrix,kmeans,"clusters_nmf_scaled.png")
hlp.getStats(labels,kmeans.labels_)

print "--------------Logarithmic NMF----------------"
nmf_matrix = nmf.fit_transform(tfidf_matrix)
log_matrix = np.log(nmf_matrix+1)
kmeans.fit(log_matrix)
hlp.plotClusters(log_matrix,kmeans,"clusters_nmf_log.png")
hlp.getStats(labels,kmeans.labels_)

print "--------------Log scaled NMF----------------"
nmf_matrix = nmf.fit_transform(tfidf_matrix)
log_matrix = np.log(nmf_matrix+1)
nmf_matrix_scaled = preprocessing.scale(log_matrix, with_mean = False)
kmeans.fit(nmf_matrix_scaled)
hlp.plotClusters(nmf_matrix_scaled,kmeans,"clusters_nmf_log_scaled.png")
hlp.getStats(labels,kmeans.labels_)

print "--------------Scaled log NMF----------------"
nmf_matrix = nmf.fit_transform(tfidf_matrix)
scaled_matrix = preprocessing.scale(nmf_matrix, with_mean = False)
log_scaled_nmf = np.log(scaled_matrix+1)
kmeans.fit(log_scaled_nmf)
hlp.plotClusters(log_scaled_nmf,kmeans,"clusters_nmf_scaled_log.png")
hlp.getStats(labels,kmeans.labels_)

