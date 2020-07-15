#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 09:23:30 2020

@author: dipu
"""
import pickle
import glob
from collections import defaultdict

def pickle_save(fname, data):
    with open(fname, 'wb') as pf:
        pickle.dump(data, pf)
        print('Saved to {}.'.format(fname))

def pickle_load(fname):
    with open(fname, 'rb') as pf:
         data = pickle.load(pf)
         print('Loaded {}.'.format(fname))
         return data

segments = glob.glob('iouValues_segment1000_*.pkl')

all_seg = defaultdict(dict)

for seg in segments:
    temp = pickle_load(seg) 
    all_seg.update(temp)

all_seg = dict(all_seg)

pickle_save('iouValues_combined_48K.pkl', all_seg)

