# -*- coding: utf-8 -*-
"""
@author: samujjwal
"""

import os
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0,parentdir)
import pytest
from mywork import task2


def test_LSI():
    task2.run_with_LSI('../data/deals.txt','../output/task2_with_lsi.txt')

def test_LDA():
    task2.run_with_LSI('../data/deals.txt','../output/task2_with_lda.txt')
    
def test_HDPLDA():
    task2.run_with_LSI('../data/deals.txt','../output/task2_with_hdplda.txt')