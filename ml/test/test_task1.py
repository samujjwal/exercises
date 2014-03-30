# -*- coding: utf-8 -*-
"""
@author: samujjwal
"""

import os
from collections import Counter
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0,parentdir)
import pytest
from mywork import task1

#uncomment the line below if the system doesnot have nltk data downloaded (only for the first execution)
#nltk.download()

counter=Counter()
counter1=Counter()
def test_create_counter():
    datafile='testdata_task1.txt'
    udatafile='testdataunique_task1.txt'
    counter=task1.create_counter(datafile)
    counter1=task1.create_counter(udatafile)
    print counter
    print "\n\n"
    print counter1
    print '----------------------------------'    
    
    assert(len(counter)==len(counter1))
    
    print "\n\n"
    counter=task1.create_counter(datafile,stop_words=[])
    counter1=task1.create_counter(udatafile,stop_words=[])
    print counter
    print "\n\n"
    print counter1
    print '----------------------------------'

    assert(len(counter)==len(counter1))
    
    print "\n\n"
    counter=task1.create_counter(datafile,punctuation="")
    counter1=task1.create_counter(udatafile,punctuation="")
    print counter
    print "\n\n"
    print counter1    
    print '----------------------------------'
    assert(len(counter)==len(counter1))
    
    print "\n\n"
    counter=task1.create_counter(datafile,punctuation="",stop_words=[])
    counter1=task1.create_counter(udatafile,punctuation="",stop_words=[])
    print counter
    print "\n\n"
    print counter1
    assert(len(counter)==len(counter1))
    print '----------------------------------'
    

# Uncomment to print output    
test_create_counter()
