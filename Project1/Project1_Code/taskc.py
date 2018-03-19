import helper as hlp
import numpy as np
import math
import operator
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction import text

class StemTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.regex_tokenizer = RegexpTokenizer(r'\w+')
        
    def __call__(self, doc):
        tmp = [self.stemmer.stem(t) for t in self.regex_tokenizer.tokenize(doc)]
        return tmp
    
all_categories=hlp.fetch_all_categories()
total_documents_per_category=[]
stop_words = text.ENGLISH_STOP_WORDS

for category in all_categories:
    categories=[category]
    all_data = fetch_20newsgroups(subset='train',categories=categories).data
    temp = ""
    for doc in all_data:
        temp= temp + " "+doc
    total_documents_per_category.append(temp)

def calculate_tcicf(freq, maxFreq, categories, categories_per_term):
    val= freq*(math.log10(categories/float(categories_per_term))+1)
    return val

def calculate():

    max_term_freq_per_category=[0]*vectorized_newsgroups_train.shape[0]
    category_count_per_term=[0]*vectorized_newsgroups_train.shape[1]

    for i in range(0,vectorized_newsgroups_train.shape[0],1):
        max_term_freq_per_category[i]=np.amax(vectorized_newsgroups_train[i,:])

    for i in range(0,vectorized_newsgroups_train.shape[1],1):
        for j in range(0,vectorized_newsgroups_train.shape[0],1):
                category_count_per_term[i]+= (0 if vectorized_newsgroups_train[j,i]==0 else 1)

    # Calculate tc-icf - Notice the matrix is sparse!
    # print len(vectorizer.get_feature_names())

    tf_icf = np.zeros((len(vectorizer.get_feature_names()), vectorized_newsgroups_train.shape[1]))

    for i in range(vectorized_newsgroups_train.shape[1]):
        row = vectorized_newsgroups_train[:,i].toarray()
        for j in range(vectorized_newsgroups_train.shape[0]):
            # print row[j,0],max_term_freq_per_category[j],len(all_categories),category_count_per_term[i]
            tf_icf[i][j] = calculate_tcicf(row[j,0],max_term_freq_per_category[j],len(all_categories),category_count_per_term[i])

    return tf_icf

for minDF in [2,5]:
    vectorizer = vectorizer = CountVectorizer(analyzer='word',stop_words=stop_words,ngram_range=(1, 1), tokenizer=StemTokenizer(), lowercase=True, max_df=0.99, min_df=minDF)
    vectorized_newsgroups_train = vectorizer.fit_transform(total_documents_per_category)

    tf_icf=calculate()
    
    print "For min_df = ", minDF
    for category in [2,3,14,15]:
        tficf={}
        term_index=0;
        for term in vectorizer.get_feature_names():
            tficf[term]=tf_icf[term_index][category]
            term_index+=1
        significant_terms = dict(sorted(tficf.iteritems(), key=operator.itemgetter(1), reverse=True)[:10]) #get 10 significant terms
        print "Significant terms for ", all_categories[category], " = ", significant_terms.keys()
    
    

