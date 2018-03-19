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

class StemTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.regex_tokenizer = RegexpTokenizer(r'\w+')
        
    def __call__(self, doc):
        tmp = [self.stemmer.stem(t) for t in self.regex_tokenizer.tokenize(doc)]
        return tmp

def fetch_categories():
    categories = ['comp.graphics', 'comp.sys.mac.hardware', 'comp.sys.ibm.pc.hardware', 'comp.os.ms-windows.misc', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
    return categories

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
    return CountVectorizer(analyzer='word',stop_words=stop_words,ngram_range=(1, 1), tokenizer=StemTokenizer(),lowercase=True, min_df=minDFVal)

def get_TFIDFTransformer():
    return TfidfTransformer()

def getSVD():
    return TruncatedSVD(n_components=50, n_iter=7, random_state=42)

def getNMF():
    return NMF(n_components=50, init='random', random_state=0)

def fetch_data():
    train_data = fetch_20newsgroups(subset='train', categories=fetch_categories(), shuffle=True, random_state=42)
    test_data = fetch_20newsgroups(subset='test', categories=fetch_categories(), shuffle=True, random_state=42)
    return train_data, test_data

def classify_into_two_class(inputData):
    for i in range(len(inputData.target)):
        inputData.target[i]=0 if inputData.target[i]<4 else 1

def getStats(actual, predicted):
    print "Accuracy : ", smet.accuracy_score(actual, predicted) * 100
    print "Precision : ", smet.precision_score(actual, predicted, average='macro') * 100

    print "Recall : ", smet.recall_score(actual, predicted, average='macro') * 100

    print "Confusion Matrix : ", smet.confusion_matrix(actual, predicted)

def plot_roc(actual, predicted, classifier_name):
    x, y, _ = roc_curve(actual, predicted)
    plt.plot(x, y, label="ROC Curve")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.2])

    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(classifier_name)
    plt.legend(loc="best")
    plt.show()
