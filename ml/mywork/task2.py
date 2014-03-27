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
from gensim import corpora

def parseLine(line, stopWords_, wordInd, currWrd):
    """ Removes stop words and lemmas using nltk and punctuations 
    using re. Returns a list with valid words in the line. currWrd is
    the index of next word occurring for the first time
    """
    lineWords = []
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
        # Update lineWords with word.
        lineWords.append(word)
    return lineWords

def parseFile(fileName, wordInd):
    """ Read words from file line by line and call parseLine()."""    
    # Get English stop words from nltk.
    stopWords_ = stopwords.words('english')    
    # Words online, buy, and shop are not relevant in this context.
    stopWords_.extend(['online', 'shop', 'buy'])
    stopWords_ = set(stopWords_)    # set gives faster access
    fileWords = []    # list of list stores words in each line
    currWrd = [0]    # currWrd is the index of next word
    with open(fileName,'r') as fIn:
        # Read each line in file. Extract words from line.
        for line in fIn:
            lineWords = parseLine(line, stopWords_, wordInd, currWrd)
            fileWords.append(lineWords)
    return fileWords

def getLdaEnt(corpus, id2w, k):
    """ Generate LDA model with k topics and returns sum of all topic 
    entropys.
    """    
    model = lda.LdaModel(corpus, id2word=id2w, num_topics=k, chunksize=10000,
                         passes=2)    
    # get the topn words in the k topics from the LDA model
    topnNum = 100
    tops = model.show_topics(topics=k, topn=topnNum, formatted = False)
    # Compute entropy of each topic
    entropy = []
    for i in range(0,k):
        entropy.append(0) 
        for j in range(0,topnNum):
            p = tops[i][j][0]
            entropy[i] -= p*math.log(p, 2)          
    return entropy

def getAllEnt(corpus, id2w, kVals):
    """ Generate LDA model with k topics for all k in kVals
    and returns list of sum of all topic entropys.
    """     
    allEnt = []
    for k in kVals:
        ent = getLdaEnt(corpus,id2w, k)
        # display statistics to help in tuning
        print k, np.mean(ent), min(ent), max(ent), np.std(ent)
        allEnt.append(np.mean(ent))
    return allEnt

def getDocClust(doc2Top, k):
    """ Compute cluster membership of document based on topic with
    maximum probability. Return a list of tuples 
    (document number, cluster membership probability)
    """    
    # Initalize list of document numbers to mark cluster membership.    
    docClust = []    
    for i in range(0,k):
        docClust.append([])
    # Assign document indices to one of k clusters, based on index of
    # topic with maximum probability    
    docNum = 0;
    for doc in doc2Top:
        maximum = doc[0][1]
        index = 0
        for i in range(1,k):
            if doc[i][1] > maximum:
                maximum = doc[i][1]
                index = i
        docClust[index].append((docNum,maximum))
        docNum += 1
    return docClust

def dClustComp(item1, item2):
    """ Comparison function for sorted() on list of document
    cluster membership. Items are tuples 
    (document number, cluster membership probability)
    """     
    if item1[1] < item2[1]:
        return 1
    elif item1[1] > item2[1]:
        return -1
    else:
        return 0
            
if __name__== '__main__':
    wordInd = {} 
    fileName = '../data/deals.txt'
    fileWords = parseFile(fileName, wordInd)
    # Create Dictionary using fileWords
    id2w = corpora.Dictionary(fileWords)
    # Creates the Bag of Word corpus.
    corpus = [id2w.doc2bow(line) for line in fileWords]
    # Ideal number of topics identified as 10 based on plot obtained
    # finalfig.png, and the entropy values and also by 
    # visually inspecting the resulting topics
    k = 10 
    model = lda.LdaModel(corpus, id2word=id2w, num_topics=k, chunksize=10000,
                         passes=2)    
    # get the topn words in the k topics from the LDA model
    topnNum = 10
    tops = model.show_topics(topics=k, topn=topnNum, formatted = False)
    # Display the top 10 terms in each of the 10 topic vectors
    print "The top ten terms in the", k, "topics are:" 
    for i in range(0,k):
        print "\t",
        for j in range(0,topnNum):
            print tops[i][j][1],
        print
    # Get document-topic matrix using model
    doc2TopIter = model[corpus]
    doc2Top =  [doc for doc in doc2TopIter]
    docClust = getDocClust(doc2Top, k)
    # The groups of deals are printed below. Due to the large number
    # of deals, only 10 deals are displayed per group. Displayed deals
    # have largest probability of membership in the topic cluster
    for i in range(0,k):
        sorted(docClust[index], cmp=dClustComp)
        print "\n\nSelected deals in group", i + 1, "are:"
        for j in range(0,10):
            docNum = docClust[i][j][0]
            print "\t",' '.join(fileWords[docNum])  