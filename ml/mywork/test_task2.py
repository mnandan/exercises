import pytest
import task2

def testparseLine1():
    w = {}
    stopWords = {}
    currWrd = [0]
    line = "The wi-fi Credit Pros credit"
    lw = task2.parseLine(line, stopWords, w, currWrd)    
    assert(len(w) == 4)
    assert(len(lw) == 5)
    assert(w['credit'] == 2)    
    assert(lw[1] == 'wifi')
    assert(lw[4] == 'credit')    

def testparseFile():
    w = {}
    fileName = 'testPF2.dat'
    flWrd = task2.parseFile(fileName, w)
    assert(len(w) == 19)
    assert(len(flWrd) == 4)
    assert(w['train'] == 5)
    assert(w['guitar'] == 18)    
        
    assert(flWrd[0][0] == 'credit')
    assert(flWrd[2][1] == 'train')
    assert(flWrd[3][-1] == 'guitar')

    assert(len(flWrd[0]) == 7)    
    assert(len(flWrd[1]) == 14)
    assert(len(flWrd[2]) == 14)
    assert(len(flWrd[3]) == 7)

def testgetDC():
    doc2Top = [[(0,0.4),(1,0.3),(2,0.3)], [(0,0.3),(1,0.2),(2,0.5)],
               [(0,0.2),(1,0.45),(2,0.35)], [(0,0.1),(1,0.8),(2,0.1)]]  
    docClust = task2.getDocClust(doc2Top, 3)
    assert(len(docClust) == 3)
    assert(len(docClust[0]) == 1)
    assert(docClust[0][0] == (0,0.4))    
    assert(len(docClust[1]) == 2)
    assert(docClust[1][0] == (2,0.45))    
    assert(docClust[1][1] == (3,0.8))    
    assert(len(docClust[2]) == 1)
    assert(docClust[2][0] == (1,0.5))    
    
def testCComp():
    item1 = (1,0.4)
    item2 = (9,0.1)    
    ret = task2.dClustComp(item1, item2)
    assert(ret == -1)
    item3 = (9,0.4)    
    ret = task2.dClustComp(item1, item3)
    assert(ret == 0)
    item4 = (9,0.7)    
    ret = task2.dClustComp(item1, item4)
    assert(ret == 1)
    a = [item1, item2, item3, item4]
    a = sorted(a,cmp=task2.dClustComp)
    assert(a[0] == item4)
    assert(a[1] == item3 or a[1] == item1)
    assert(a[2] == item3 or a[2] == item1)
    assert(a[3] == item2)