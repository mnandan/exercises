import pytest
import task3

def testparseLine1():
    w = {}
    stopWords = {}
    currWrd = [0]
    line = "The wi-fi Credit Pros credit"
    lw = task3.parseLine(line, stopWords, w, currWrd)    
    assert(len(w) == 4)
    assert(len(lw) == 4)
    assert(w['credit'] == 2)    
    assert(lw[w['credit']] == 2)
    assert(lw[w['wifi']] == 1)    

def testparseFile1():
    w = {}
    fileNames = [('testPF2.dat',0),('testPF2.dat',1)]
    (flWrd,Y) = task3.parseFiles(fileNames, w)
    assert(len(w) == 19)
    assert(len(flWrd) == 8)
    assert(w['train'] == 5)
    assert(w['guitar'] == 18)    
 
    assert(len(flWrd[0]) == 4)    
    assert(len(flWrd[5]) == 14)
    assert(len(flWrd[2]) == 14)
    assert(len(flWrd[7]) == 5)
 
def testparseFile2():
    w = {}
    fileNames = [('testPF2.dat',0),('testPF2.dat',1)]
    (flWrd,Y) = task3.parseFiles(fileNames, w)
        
    assert(flWrd[0][w['credit']] == 3)
    assert(flWrd[2][w['train']] == 1)
    assert(flWrd[6][w['train']] == 1)    
    assert(flWrd[3][w['guitar']] == 3)
    assert(flWrd[7][w['guitar']] == 3)
    assert(sum(Y[:3]) == 0)
    assert(sum(Y[4:]) == 4) 

def testTFIDF():
    w = {}
    fileNames = [('testPF2.dat',0),('testPF2.dat',1)]
    X,Y = task3.getTFIDF(fileNames, w)
    assert(X.shape == (8,19))
    assert(X[4].nnz == 4)
    assert(X[1].nnz == 14)
    assert(X[6].nnz == 14)
    assert(X[3].nnz == 5)