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


class MyTokenizer(object):
    def __init__(self):
        self.regex_tokenizer = RegexpTokenizer(r'\w+')
        
    def __call__(self, doc):
        return self.regex_tokenizer.tokenize(doc)

def fetch_categories():
    categories = ['comp.graphics', 'comp.sys.mac.hardware', 'comp.sys.ibm.pc.hardware', 'comp.os.ms-windows.misc', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
    return categories

def fetch_labels(dataset):
    return dataset.target

def fetch_all_categories():
    all_categories=['comp.graphics',
                'comp.os.ms-windows.misc',
                'comp.sys.ibm.pc.hardware',
                'comp.sys.mac.hardware',
                'comp.windows.x',
                'rec.autos',
                'rec.motorcycles',
                'rec.sport.baseball',
                'rec.sport.hockey',
                'alt.atheism',
                'sci.crypt',
                'sci.electronics',
                'sci.med',
                'sci.space',
                'soc.religion.christian',
                'misc.forsale',
                'talk.politics.guns',
                'talk.politics.mideast',
                'talk.politics.misc',
                'talk.religion.misc'
                ]
    return all_categories

def get_CountVectorizer(minDFVal):
    stop_words = text.ENGLISH_STOP_WORDS
    return CountVectorizer(analyzer='word',stop_words=stop_words, tokenizer=MyTokenizer(),lowercase=True, min_df=minDFVal)

def get_TFIDFTransformer():
    return TfidfTransformer()

def getSVD(r):
    return TruncatedSVD(n_components=r, random_state=42)

def getNMF(r):
    return NMF(n_components=r, init='random', random_state=0)

def getKmeans(num):
    return KMeans(n_clusters=num, init='k-means++', max_iter=300, n_init=10, random_state=42)

def fetch_data():
    dataset = fetch_20newsgroups(subset='all', categories=fetch_categories(), shuffle=True, random_state=42)
    return dataset

def fetch_all_data():
    dataset = fetch_20newsgroups(subset='all', categories=fetch_all_categories(), shuffle=True, random_state=42)
    return dataset

def classify_into_two_class(inputData):
    for i in range(len(inputData.target)):
        inputData.target[i]=0 if inputData.target[i]<4 else 1
        
def getStats(actual, predicted):
    homo = smet.homogeneity_score(actual, predicted)
    comp = smet.completeness_score(actual, predicted)
    vscore = smet.v_measure_score(actual, predicted)
    adjscore = smet.adjusted_rand_score(actual, predicted)
    infoscore = smet.adjusted_mutual_info_score(actual, predicted)
    print ("Homogeniety : %0.3f" % homo)
    print ("Completeness : %0.3f" % comp)
    print ("V-measure : %0.3f" % vscore)
    print ("Adjusted Rand Score : %0.3f" % adjscore)
    print ("Adjusted Mutual Info Score : %0.3f" % infoscore)
    printContingencyTable(actual,predicted)
    return homo, comp, vscore, adjscore, infoscore

def printContingencyTable(actual,predicted):
    print "Contigency Table : \n"
    mat = smet.confusion_matrix(actual, predicted)
    print mat
    plt.matshow(mat)
    plt.show()
    
    
def plot20Clusters(X,kmeans,name):
    colors=["red","green","blue","yellow","orange","cyan","purple","black","lightblue","lime","grey","lightgreen","pink","magenta","darkgreen","aqua","coral","turquoise","brown","blueviolet"]
    for i in range(0,20):
        x1 = X[kmeans.labels_ == i][:, 0]
        y1 = X[kmeans.labels_ == i][:, 1]
        plt.plot(x1, y1, colors[i], marker='+', linestyle=' ')
    
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
    marker='x', s=169, linewidths=3,
    color='w', zorder=10)
    plt.title(name)
    plt.savefig("plots/"+name, format='png')
    plt.show()
    
    
def plotClusters(X,kmeans,name):
	x1 = X[kmeans.labels_ == 0][:, 0]
	y1 = X[kmeans.labels_ == 0][:, 1]
	plt.plot(x1, y1, 'r+')

	x2 = X[kmeans.labels_ == 1][:, 0]
	y2 = X[kmeans.labels_ == 1][:, 1]
	plt.plot(x2, y2, 'g+')

	centroids = kmeans.cluster_centers_
	plt.scatter(centroids[:, 0], centroids[:, 1],
			marker='x', s=169, linewidths=3,
			color='w', zorder=10)
			
	plt.savefig("plots/"+name, format='png')
	plt.show()