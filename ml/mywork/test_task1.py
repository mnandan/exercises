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
        
def testUScr():
    w = {}
    g = {}
    stopWords = {}
    line = "The wi_fi _Credit Pros"
    task1.parseLine(line, stopWords, w, g)    
    assert(len(w) == 5)
    assert('wi_fi' not in w)
    assert('wi' in w)
    assert('fi' in w)    
    assert('credit' in w)
        
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
    stopWords = set(['the', 'and'])
    line = "The Credit Pros and Credit Reports  in pages 123 - 0x55"
    task1.parseLine(line, stopWords, w, g)    
    assert(len(w) == 5)
    assert('123' not in w)
    assert('0x55' not in w)    
        
def testWC():
    w = {}
    g = {}
    stopWords = set(['the', 'and', 'is'])
    line = "The Credit Pros and Credit Reports in pages 123 - 0x55. Credit is not reported"
    task1.parseLine(line, stopWords, w, g)    
    assert(len(w) == 6)
    assert(w['credit'] == 3)
    assert(w['report'] == 2)    
    assert(w['page'] == 1)
    assert(w['in'] == 1)

def testGtr():
    w = {}
    g = {}
    stopWords = set(['the', 'and'])
    line = "The Credit guitar and Reports guitar are not reported. The guitar is nice"
    task1.parseLine(line, stopWords, w, g)    
    assert(len(g) == 2)
    assert('credit' in g)
    assert('report' in g)  
    
def testGtr2():
    w = {}
    g = {}
    stopWords = set(['the', 'and'])
    line = "The Credit guitars and Reports guitars are not reported. The guitar is nice"
    task1.parseLine(line, stopWords, w, g)    
    assert(len(g) == 2)
    assert('credit' in g)
    assert('report' in g)
    
def testPF1():
    w = {}
    g = {}
    fileName = 'testPF1.dat'
    task1.parseFile(fileName, w, g)
    assert(len(w) == 18)
    assert(len(g) == 0)

def testPF2():
    w = {}
    g = {}
    fileName = 'testPF2.dat'
    task1.parseFile(fileName, w, g)
    assert('guitars' not in w)
    assert(len(w) == 19)
    assert(len(g) == 3)
    
def testSC():    
    w = {}
    g = {}
    fileName = 'testPF1.dat'
    (maxTerm, maxCnt, minTerm, minCnt) = task1.smartCount(fileName, w, g)
    assert('credit' == maxTerm)
    assert(maxCnt == 3)
    assert('page' == minTerm)    
    assert(minCnt == 1)
       
                     