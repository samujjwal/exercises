# -*- coding: utf-8 -*-
"""
@author: samujjwal
"""

import os
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0,parentdir)
import pytest
from mywork import task3


def test_clasify():
    gooddata='../data/good_deals.txt'
    baddata='../data/bad_deals.txt'
    testdata='../data/test_deals.txt'
    task3.classify_and_writetofile(gooddata,baddata,testdata)
