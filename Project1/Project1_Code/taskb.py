import helper as hlp

twenty_train, twenty_test = hlp.fetch_data()


def getTFIDF_matrix(trainData, testData, minDFVal):
    count_vect = hlp.get_CountVectorizer(minDFVal)
    count_train=count_vect.fit_transform(trainData.data)
    transformer = hlp.get_TFIDFTransformer()
    count_train_tfidf=transformer.fit_transform(count_train)
    
    count_test=count_vect.transform(testData.data)
    count_test_tfidf=transformer.transform(count_test)
    
    return count_train_tfidf, count_test_tfidf
    
for min_df in [2,5]:    
    train_tfidf, test_tfidf=getTFIDF_matrix(twenty_train,twenty_test,min_df)
    print "Number of terms extracted using TFIDF vector representation with min_df=", min_df ,"is ", train_tfidf.shape[1]

