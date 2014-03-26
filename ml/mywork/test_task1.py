import pytest
import task1

def testHypenated():
    w = {}
    g = {}
    stopWords = {}
    line = "The wi-fi Credit Pros"
    task1.parseLine(line, stopWords, w, g)    
    assert(len(w) == 4)
    assert('wi-fi' not in w)
    assert('wifi' in w)
        
def testPunctuation1():
    w = {}
    g = {}
    stopWords = set()
    line = "Clean - Up Yours Credit Report"
    task1.parseLine(line, stopWords, w, g)    
    assert('-' not in w)
    assert(len(w) == 5)

def testPunctuation12():
    w = {}
    g = {}
    stopWords = {}
    line = "Legal Credit \"Repair\""
    task1.parseLine(line, stopWords, w, g)    
    assert('"' not in w)
    assert(len(w) == 3)    
        
def testPunctuation3():
    w = {}
    g = {}
    stopWords = set()
    line = "Clean  Up Your's Credit Report's pages."
    task1.parseLine(line, stopWords, w, g) 
    assert(len(w) == 6)   
    for word in w:
        assert("'" not in word)
        
def testPunctuation4():
    w = {}
    g = {}
    stopWords = set()
    line = "The Credit Pros, Clean - Up Your Credit Report. Legal, Credit Repair"
    task1.parseLine(line, stopWords, w, g)    
    assert(len(w) == 9)   
    for word in w:
        assert(',' not in word)
        
def testStopWrd():
    w = {}
    g = {}
    stopWords = set(['the', 'and', 'be'])
    line = "The Credit Pros and Credit Reports  in pages be Legal, Credit"
    task1.parseLine(line, stopWords, w, g)    
    assert(len(w) == 6)
    assert('The' not in w)
    assert('the' not in w)    
    assert('and' not in w)    
    assert('be' not in w)   
    

def testNumbers():
    w = {}
    g = {}
    stopWords = set(['the', 'and', 'be'])
    line = "The Credit Pros and Credit Reports  in pages 123 - 0x55"
    task1.parseLine(line, stopWords, w, g)    
    assert(len(w) == 5)
    assert('123' not in w)
    assert('0x55' not in w)    
        
def testWC():
    w = {}
    g = {}
    stopWords = set(['the', 'and', 'be'])
    line = "The Credit Pros and Credit Reports in pages 123 - 0x55. Credit is not reported"
    task1.parseLine(line, stopWords, w, g)    
    assert(len(w) == 6)     #is and be are lemmas
    assert(w['credit'] == 3)
    assert(w['report'] == 2)    
    assert(w['page'] == 1)
    assert(w['in'] == 1)
                     