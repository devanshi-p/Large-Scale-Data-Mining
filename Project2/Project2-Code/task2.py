import helper as hlp
import task1 as t1

dataset = hlp.fetch_data()
hlp.classify_into_two_class(dataset)
labels = hlp.fetch_labels(dataset)

tfidf_matrix = t1.getTFIDF_matrix(dataset,3)
km = hlp.getKmeans(2)
km.fit(tfidf_matrix)
hlp.getStats(labels,km.labels_)
