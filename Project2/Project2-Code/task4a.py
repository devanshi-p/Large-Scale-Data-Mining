import helper as hlp
import task1 as t1
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


dataset = hlp.fetch_data()

hlp.classify_into_two_class(dataset)
labels = hlp.fetch_labels(dataset)

tfidf_matrix = t1.getTFIDF_matrix(dataset,3)
kmeans = hlp.getKmeans(2)

svd = hlp.getSVD(3)
svd_matrix = svd.fit_transform(tfidf_matrix)
kmeans.fit(svd_matrix)
hlp.plotClusters(svd_matrix,kmeans,"clusters_2d_svd_best.png")
hlp.getStats(labels,kmeans.labels_)

nmf = hlp.getNMF(10)
nmf_matrix = nmf.fit_transform(tfidf_matrix)
kmeans.fit(nmf_matrix)
hlp.plotClusters(nmf_matrix,kmeans,"clusters_2d_nmf_best.png")
hlp.getStats(labels,kmeans.labels_)


