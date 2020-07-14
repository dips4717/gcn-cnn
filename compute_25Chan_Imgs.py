#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 11:48:01 2020

@author: dipu
"""

import pickle
import torch
import os
from multiprocessing import Pool, Value
import numpy as np
import torch.nn.functional as F


class Counter(object):
    def __init__(self):
        self.val = Value('i', 0)

    def add(self, n=1):
        with self.val.get_lock():
            self.val.value += n

    @property
    def value(self):
        return self.val.value

def get_classwise_channel_image(id):
    counter.add(1)
    num_class = 25  # Num of channels = num of classes
    temp_info = info[id]
    
    W, H = 1440, 2560
    c_img = torch.zeros(num_class, H, W)  # C*H*W
    
    class_id = temp_info['class_id']
    n_comp = len(temp_info['class_id'])
    
    for i in range(n_comp):
        x1, y1, w, h = temp_info['xywh'][i]
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x1+w)
        y2 = int(y1+h)
        channel = class_id[i]-1
        #c_img[channel, x1:x2+1, y1:y2+1] = 1
        c_img[channel, y1:y2, x1:x2+1  ] =1
    
    c_img = c_img.unsqueeze(0)
    c_img = F.interpolate(c_img, size= [254, 126])
    c_img = c_img.squeeze(0)
    
    c_img = c_img.numpy()    
    c_img = c_img.astype(bool)
    #    np.savez_compressed(os.path.join(save_dir, str(id)), im = c_img)
    
    if counter.value % 100 == 0 and counter.value >= 100:
        print('{} / {}'.format(counter.value, len(info.keys())))  
    
    np.save( save_dir + '%s'%(id), c_img)
    
    #torch.save(c_img, os.path.join(save_dir, str(id) + '.pt'))
    #sys.getsizeof
    
    
    
save_dir = 'data/25ChanImages/'
if os.path.exists(save_dir):
    raise Exception("dir already exists")
else:
    os.mkdir(save_dir) 

with open('data/rico_box_info.pkl', 'rb') as f:
    info = pickle.load(f)  
ids = list(info.keys())     
counter = Counter()

p = Pool(20)
print("[INFO] Start")
results = p.map(get_classwise_channel_image, info.keys())
print("Done")


#for i in range (1):
#    get_classwise_channel_image(ids[i])