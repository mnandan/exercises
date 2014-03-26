""" Features

The objective of this task is to explore the corpus, deals.txt. 

The deals.txt file is a collection of deal descriptions, separated by a new line, from which 
we want to glean the following insights:

1. What is the most popular term across all the deals?
2. What is the least popular term across all the deals?
3. How many types of guitars are mentioned across all the deals?

"""
import sys
from nltk.corpus import stopwords
import re
from nltk.stem.wordnet import WordNetLemmatizer
import nltk

def parseLine(line, stopWords_, wordCnt, gitrType):
    """Stores a dict wordCnt with word as key and count as value. If a 
    guitar is found its type (the word preceding it) is stored in dict
    gitrType. Removes stop words and lemmas using nltk. Also removes
    punctuations.
    """
    prevWord = ''
    # Hypen in hyphenated words are removed e.g. wi-fi ==> wifi.
    line = re.sub('(\w)-(\w)',r'\1\2',line)
    # replace underscore with space     
    line = re.sub('(\w)_(\w)',r'\1 \2',line)    
    # Remove punctuation marks.
    line = re.sub("[',~`@#$%^&*|<>{}[\]\\\/.:;?!\(\)_\"-]",r'',line)
    wnLmtzr = WordNetLemmatizer()
  
    for word in line.split():
        # increment count of word in wordCnt. If it is seen for the 
        # first time add it to wordCnt
        word = word.lower()    # case of words is ignored
        # Lemmatize word using word net function
        word = wnLmtzr.lemmatize(word, 'n')    # with noun
        word1 = wnLmtzr.lemmatize(word, 'v')    # with verb
        if len(word1) < len(word):    # select smaller of two
            word = word1                
        # Ignore stop words and numbers.
        if word in stopWords_ or \
                re.match('^\d+x?\d*$',word) is not None:
            prevWord = ''
            continue
        # Update wordCnt with number of occurrences of word.
        if word in wordCnt:                
            wordCnt[word] += 1
        else:
            wordCnt[word] = 1
        # If word is guitar store the type in gitrType.
        if word == 'guitar':
            if prevWord != '':    # previous word not stop word
                gitrType[prevWord] = 1
        prevWord = word
                    
def parseFile(fileName, wordCnt, gitrType):
    """ Read words from file line by line and call parseLine()."""    
    # Get English stop words from nltk.
    stopWords_ = stopwords.words('english')    
    # Words online, buy, and shop are not relevant in this context.
    stopWords_.extend(['online', 'shop', 'buy'])
    stopWords_ = set(stopWords_)    # set gives faster access
    with open(fileName,'r') as fIn:
        # Read each line in file. Extract words from line.
        for line in fIn:
            parseLine(line, stopWords_, wordCnt, gitrType)


def smartCount(fileName, wordCnt, gitrType):
    """ Calls parseFile() to get word count and guitar count. Finds 
    most popular and least popular terms and also the number of guitar
    types. 
    """ 
    parseFile(fileName, wordCnt, gitrType)
    # find most popular and least popular terms
    maxCnt = 0;
    maxTerm = '';
    minCnt = sys.maxint;
    minTerm = '';
    for word in wordCnt:
        if(wordCnt[word] > maxCnt):
            maxCnt = wordCnt[word]
            maxTerm = word
        elif(wordCnt[word] < minCnt):
            minCnt = wordCnt[word]
            minTerm = word
    
    print "Results of word count:"
    print "  most popular term = ", maxTerm
    print "  least popular term = ", minTerm
    print "  number of types of guitars = ", len(gitrType)
    
    return (maxTerm, maxCnt, minTerm, minCnt)
    
if __name__== '__main__':
    wordCnt = {}    # stores count of each word
    gitrType = {}    # stores all guitar types
    fileName = '../data/deals.txt'
    smartCount(fileName, wordCnt, gitrType)