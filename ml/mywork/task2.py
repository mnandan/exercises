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
from gensim.models import ldamodel as lda
import gensim.matutils as genCorp 
import matplotlib.pyplot as plt
import math
import numpy as np

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

def parseFile(fileName, wordInd):
    """ Read words from file line by line and call parseLine()."""    
    # Get English stop words from nltk.
    stopWords_ = stopwords.words('english')    
    # Words online, buy, and shop are not relevant in this context.
    stopWords_.extend(['online', 'shop', 'buy'])
    stopWords_ = set(stopWords_)    # set gives faster access
    fileWords = []    # list of dict stores count of words in each line
    currWrd = [0]    # currWrd is the index of next word
    with open(fileName,'r') as fIn:
        # Read each line in file. Extract words from line.
        for line in fIn:
            lineWords = parseLine(line, stopWords_, wordInd, currWrd)
            fileWords.append(lineWords)
    return fileWords

def getCSC(fileName, wordInd):
    """ Calls parseFile() to get wordInd, a dict with all words and 
    fileWords a list of dict that stores count of words in each line.
    Generates a sparse matrix using fileWords.
    """
    fileWords = parseFile(fileName, wordInd)
    # create CSC matrix of dimensions (Num. of deals) x (Num. of words)
    rowNum = 0
    rowList = []
    ColumnList = []
    data = []
    for lWords in fileWords:
        for colInd in lWords:
            rowList.append(rowNum)
            ColumnList.append(colInd)
            data.append(lWords[colInd])            
        rowNum += 1        
    X = sparseMat.csc_matrix((data, (rowList, ColumnList)), 
                             shape=(rowNum,len(wordInd)), dtype='int32')
    # Convert the count matrix to a matrix of TFIDF values    
    #X = TfidfTransformer().fit_transform(X)
    return X

def getLdaEnt(corpus, id2w, k):
    """ Generate LDA model with k topics and returns sum of all topic 
    entropys.
    """    
    model = lda.LdaModel(corpus, id2word=id2w, num_topics=k, chunksize=10000,
                         passes=2)    
    # get the topn words in the k topics from the LDA model
    topnNum=100
    tops = model.show_topics(topics=k, topn=topnNum, formatted = False)
    # Compute entropy of each topic
    entropy = []
    for i in range(0,k):
        entropy.append(0) 
        for j in range(0,topnNum):
            p = tops[i][j][0]
            entropy[i] -= p*math.log(p, 2)          
    return entropy

def getAllEnt(X, w, kVals):
    """ Generate LDA model with k topics for all k in kVals
    and returns list of sum of all topic entropys.
    """     
    allEnt = []
    # generate gensim corpus
    corpus = genCorp.Sparse2Corpus(X, documents_columns=False)
    # generate a hash with key and value of w reversed    
    id2w = dict((value,key) for key,value in w.iteritems())
    for k in kVals:
        ent = getLdaEnt(corpus,id2w, k)
        # display statistics to help in tuning
        print k, np.mean(ent), min(ent), max(ent), np.std(ent)
        allEnt.append(np.mean(ent))
    return allEnt
        
if __name__== '__main__':
    wordInd = {} 
    fileName = '../data/deals.txt'
    X = getCSC(fileName, wordInd)
#     kVals = range(4,14,2)    # coarse set of parameters
#     allEnt = getAllEnt(X, wordInd, kVals)
#     plt.plot(kVals, allEnt)
#     plt.ylabel('Mean of entropy')
#     plt.xlabel('Number of latent topics')
#     plt.show()    
    
    # Ideal number of topics identified as 10 based on finalfig.png
    # and the high variation in the entropy values of the topics 