""" Groups and Topics

The objective of this task is to explore the structure of the deals.txt file. 

Building on task 1, we now want to start to understand the relationships to help us understand:

1. What groups exist within the deals?
2. What topics exist within the deals?

"""
import sys
from nltk.corpus import stopwords
import re
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
import scipy.sparse as sparseMat 
from sklearn.feature_extraction.text import TfidfTransformer

def parseLine(line, stopWords_, wordCnt):
    """Updates wordCnt with word as key and index as value. Removes
    stop words and lemmas using nltk and punctuations using re.
    Returns a dict with valid words in the line as keys and their 
    count as values.
    """
    lineWords = {}
    # Hypen in hyphenated words are removed e.g. wi-fi ==> wifi.
    line = re.sub('(\w)-(\w)',r'\1\2',line)
    # replace underscore with space     
    line = re.sub('(\w)_(\w)',r'\1 \2',line)    
    # Remove punctuation marks.
    line = re.sub("[',~`@#$%^&*|<>{}[\]\\\/.:;?!\(\)_\"-]",r'',line)
    currWrd = 0    # index of next word occurring for the first time
    wnLmtzr = WordNetLemmatizer()    
    for word in line.split():
        # Get index of word from wordCnt. If it is seen for the first 
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
        # Update wordCnt with number of occurrences of word.
        if word not in wordCnt:                
            wordCnt[word] = currWrd
            currWrd += 1
        # Update lineWords with number of occurrences of word.
        if wordCnt[word] in lineWords:
            lineWords[wordCnt[word]] += 1
        else:
            lineWords[wordCnt[word]] = 1
    return lineWords

def parseFile(fileName, wordCnt):
    """ Read words from file line by line and call parseLine()."""    
    # Get English stop words from nltk.
    stopWords_ = stopwords.words('english')    
    # Words online, buy, and shop are not relevant in this context.
    stopWords_.extend(['online', 'shop', 'buy'])
    stopWords_ = set(stopWords_)    # set gives faster access
    fileWords = []    # list of dict stores count of words in each line
    with open(fileName,'r') as fIn:
        # Read each line in file. Extract words from line.
        for line in fIn:
            lineWords = parseLine(line, stopWords_, wordCnt)
            fileWords.append(lineWords)
    return fileWords

def makeTFIDF(fileName, wordCnt):
    """ Calls parseFile() to get wordCnt, a dict with all words and 
    fileWords a list of dict that stores count of words in each line.
    Generates a sparse TF-IDF matrix using fileWords.
    """ 
    fileWords = parseFile(fileName, wordCnt)
    # create CSR matrix of dimensions (Num. of deals) x (Num. of words)
    X = sparseMat.csr_matrix(len(fileWords),len(wordCnt))
    rowNum = 0
    for lWords in fileWords:
        for wordInd in lWords:
            X[rowNum][wordInd] = lWords[wordInd]
    tfidfTrans = TfidfTransformer()
    X = tfidfTrans.fit(X)
    
    return X
    
if __name__== '__main__':
    wordCnt = {}    # stores count of each word
    gitrType = {}    # stores all guitar types
    fileName = '../data/deals.txt'
    X = makeTFIDF(fileName, wordCnt)