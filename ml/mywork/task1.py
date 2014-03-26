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

def parseFile(fileName, wordCnt, gitrType):
    """ Read words from file line by line. Stores a dict wordCnt with 
    word as key and index as value. If a guitar is found its type (the
    word preceding it) is stored in dict gitrType
    """
    # Get English stop words from nltk.
    stopWords_ = stopwords.words('english')    
    # Words online, buy, and shop are not relevant in this context.
    stopWords_.extend(['online', 'shop', 'buy'])
    wnLmtzr = WordNetLemmatizer()
    #punc = re.compile(r'['"()!?.]')
    with open(fileName,'r') as fIn:
        # Read each line in file. Extract words from line.
        for line in fIn:
            prevWord = ''
            # Hypen in hyphenated words are removed e.g. wi-fi ==> wifi
            line = re.sub('(\w)-(\w)',r'\1\2',line)
            # Remove punctuation marks.
            line = re.sub("[',~`@#$%^&*|<>{}[\]\\\/.:;?!()\"-]",r'',line)            
            for word in line.split():
                # Get index of word from wordCnt. If it is seen for the 
                # first time assign an index to the word
                word = word.lower()    # case of words is ignored
                word = wnLmtzr.lemmatize(word, pos='v')    # wordnet lemmatizer                 
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
                # word is guitar store the type in gitrType
                if word == 'guitar' or word == 'guitars':
                    if prevWord != '':    # previous word not stop word
                        gitrType[prevWord] = 1
                prevWord = word

def naiveCount():
    wordCnt = {}    # stores count of each word
    gitrType = {}    # stores all guitar types
    parseFile('../data/deals.txt', wordCnt, gitrType)
    
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
    
    print "Results of naive word count:"
    print "  most popular term = ", maxTerm
    print "  least popular term = ", minTerm
    print "  number of types of guitars = ", len(gitrType)

if __name__== '__main__':
    naiveCount()