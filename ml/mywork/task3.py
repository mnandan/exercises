""" Classification

The objective of this task is to build a classifier that can tell us whether a new, unseen deal 
requires a coupon code or not. 

We would like to see a couple of steps:

1. You should use bad_deals.txt and good_deals.txt as training data
2. You should use test_deals.txt to test your code
3. Each time you tune your code, please commit that change so we can see your tuning over time

Also, provide comments on:
    - How general is your classifier?
    - How did you test your classifier?

"""
import sys
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
import re
import scipy.sparse as sparseMat 
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer 
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn import cross_validation as CV
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

def parseLine(line, stopWords_, wordInd, currWrd):
    """Updates wordInd with word as key and index as value. Removes
    stop words and lemmas using nltk and punctuations using re.
    Returns a dict with valid words in the line as keys and their 
    count as values. currWrd is the index of next word occurring for 
    the first time
    """
    lineWords = {}
    # Hypen in hyphenated words are removed e.g. wi-fi ==> wifi.
    line = re.sub('(\w)-(\w)',r'\1\2',line)
    # replace underscore with space     
    line = re.sub('(\w)_(\w)',r'\1 \2',line)    
    # Remove punctuation marks.
    line = re.sub("[',~`@#$%^&*|<>{}[\]\\\/.:;?!\(\)_+\"-]",r'',line)
    wnLmtzr = WordNetLemmatizer()    
    for word in line.split():
        # Get index of word from wordInd. If it is seen for the first 
        # time assign an index to the word.
        word = word.lower()    # case of words is ignored
        # Lemmatize word using word net function
        word = wnLmtzr.lemmatize(word, 'n')    # with noun
        word1 = wnLmtzr.lemmatize(word, 'v')    # with verb
        if len(word1) < len(word):    # select smaller of two
            word = word1                
        # Ignore stop words and numbers.
        if word in stopWords_ or \
                re.match('^\d+x?\d*$',word) is not None:
            continue
        # Update wordInd with number of occurrences of word.
        if word not in wordInd:                
            wordInd[word] = currWrd[0]
            currWrd[0] += 1
        # Update lineWords with number of occurrences of word.
        if wordInd[word] in lineWords:
            lineWords[wordInd[word]] += 1
        else:
            lineWords[wordInd[word]] = 1
    return lineWords

def parseFiles(fileNames, wordInd):
    """ Read words from filew line by line and call parseLine()."""    
    # Get English stop words from nltk.
    stopWords_ = stopwords.words('english')    
    # Words online, buy, and shop are not relevant in this context.
    stopWords_.extend(['online', 'shop', 'buy'])
    stopWords_ = set(stopWords_)    # set gives faster access
    allWords = []    # list of dict stores count of words in each line
    currWrd = [0]    # currWrd is the index of next word
    labels = []
    for file, label in fileNames:
        with open(file,'r') as fIn:
            # Read each line in file. Extract words from line.
            for line in fIn:
                lineWords = parseLine(line, stopWords_, wordInd, currWrd)
                allWords.append(lineWords)
                labels.append(label)
    return (allWords, labels)

def getTFIDF(fileNames, wordInd):
    """ Calls parseFiles() to get wordInd, a dict with all words, 
    allWords a list of dict that stores count of words in each line,
    and labels a list of the labels of each line. Generates a sparse
    TFIDF matrix using allWords.
    """
    labels = []    
    (allWords, labels) = parseFiles(fileNames, wordInd)
    # create CSR matrix of dimensions (Num. of deals) x (Num. of words)
    rowNum = 0
    rowList = []
    ColumnList = []
    data = []
    for lWords in allWords:
        for colInd in lWords:
            rowList.append(rowNum)
            ColumnList.append(colInd)
            data.append(lWords[colInd])            
        rowNum += 1        
    X = sparseMat.csr_matrix((data, (rowList, ColumnList)), 
                             shape=(rowNum,len(wordInd)), dtype='int32')
    # Convert the count matrix to a matrix of TFIDF values    
    X = TfidfTransformer().fit_transform(X)
    # Convert labels in an ndarray
    Y = np.asarray(labels).astype('int8')     
    return (X,Y)
       
if __name__== '__main__':
    wordInd = {} 
    fileNames = [('../data/bad_deals.txt',0), ('../data/good_deals.txt',1)]        
    X, Y = getTFIDF(fileNames, wordInd)
    # Make 70% of data as training set and 30% as validation set.
    X = X.todense()
    Xtr, Xts, Ytr, Yts = CV.train_test_split(X, Y, test_size=0.3,
                                              random_state=42)
    # Train a SVM on training set. Do cross validation to find 
    # best params.
    tuned_parameters = {'kernel': ['rbf'], 'gamma': [0.05], 'C': [12]}
    svmObj = SVC()
    clf = GridSearchCV(svmObj, tuned_parameters, cv=10, scoring='accuracy')
    clf.fit(Xtr, Ytr)
    predY = clf.predict(Xts)

    print "On the validation set (30% of good and bad deals) that were not",
    print "used in training, the classification accuracy is ", 
    print 100.0*clf.best_score_, "%"

    clf = SVC(kernel='rbf', gamma=0.05, C=12)
    clf.fit(X, Y)
    # Collect data from test file.
    fileNames = [('../data/test_deals.txt',2)]
    XTest, YTest = getTFIDF(fileNames, wordInd)
    XTest = XTest.todense()
    rdimT, cdimT = XTest.shape
    rdimTr, cdimTr = X.shape    
    # Deals with such words can be taken to be good.    
    markWords = ['coupon','code']
    labldCnt = 0
    for wrd in markWords:        
        if wrd in wordInd:
            colNum = wordInd[wrd]
            for i in range(0,rdimT):
                if(XTest[i,colNum] > 0):           
                    YTest[i] = 1
                    labldCnt += 1
    # If new words were discovered in test file ignore their values.   
    if cdimT > cdimTr:
        XTest = XTest[:,0:cdimTr]
    # Get results for test set.    
    predYtest = clf.predict(XTest) 
    yCnt = 0;
    for y,p in zip(YTest,predYtest):        
        if p == y:
            yCnt += 1
    print "On the automatically labeled good deals, ",
    print 100.0*float(yCnt)/labldCnt, "% deals are correctly classified"
    # Manually generated test labels.
    YTest2 = [0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 
              0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1,
              0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0]
    YTest2 = np.asarray(YTest2).astype('int8')
    # Copy previosuly identified good deals from YTest labels.    
    for i in range(0,rdimT):
        if(YTest[i] == 1):           
            YTest2[i] = 1    
    yCnt2 = 0;
    for y,p in zip(YTest2,predYtest):        
        if p == y:
            yCnt2 += 1
    print "On the manually labeled deals, ",
    print 100.0*float(yCnt2)/len(YTest2), "% deals are correctly classified"
    # Another method to estimate of accuracy on test set is with clustering.
    # Training data can be clustered and then cluster membership of test data 
    # analyzed. Labels can be assigned based on the label of the cluster 
    # it belongs to or from its neighbors.      