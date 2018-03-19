import helper as hlp
import numpy as np


dataset = hlp.fetch_data()

def getTFIDF_matrix(dataset, minDFVal):
    count_vect = hlp.get_CountVectorizer(minDFVal)
    count_transformed=count_vect.fit_transform(dataset.data)
    transformer = hlp.get_TFIDFTransformer()
    count_tfidf=transformer.fit_transform(count_transformed)
    return count_tfidf

dataset_tfidf=getTFIDF_matrix(dataset,3)
print "Dimensions of the TFIDF matrix is ", dataset_tfidf.shape
