#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 11:27:35 2020
Compute an average iou between two images.
For each image in train_set, the pairwise iou for rest of images in dataset are computed.

@author: dipu
"""

import numpy as np
import os
import pickle
from multiprocessing import Pool, Value
from collections import defaultdict 
import time
from functools import partial
import argparse
import random

class Counter(object):
    def __init__(self):
        self.val = Value('i', 0)

    def add(self, n=1):
        with self.val.get_lock():
            self.val.value += n

    @property
    def value(self):
        return self.val.value

Channel_img_dir = '../data/25ChannelImages'


split_set_file = '../data/UI_data.p'
rico_split_set = pickle.load(open(split_set_file, 'rb'))
train_uis = rico_split_set['train_uis'] 
train_uis = [x.replace('.png', '') for x in train_uis]


## Load all the images in memory, into a dictionay
img_dict = defaultdict(dict)

for ii in range(len(train_uis)):
   imgA_pth = os.path.join(Channel_img_dir, train_uis[ii] + '.npy' ) 
   imgA = np.load(imgA_pth)
   img_dict[train_uis[ii]] = imgA
   if ii%1000 == 0:
       print('Loaded', ii)

print('\nSuccessfully Loaded all the images!')


def compute_iou(imgA_id, imgB_id):
    
    #counter.add(1)
    #imgA_pth = os.path.join(Channel_img_dir, imgA_id + '.npy' )
    #imgB_pth = os.path.join(Channel_img_dir, imgB_id + '.npy' )
    
    #imgA = np.load(imgA_pth) 
    #imgB = np.load(imgB_pth)
    
    imgA = img_dict[imgA_id]
    imgB = img_dict[imgB_id]
    
    product = imgA*imgB
    inter = np.sum(product,axis=(1,2))

    logical_or = imgA + imgB
    union = np.sum(logical_or, axis= (1,2))
    
    #n_class_q = (np.sum(imgA, axis= (1,2)) > 0).sum() # number of components in first image
    n_class_union = (np.sum(union, axis= (1,2)) > 0).sum()
        
    with np.errstate(divide='ignore', invalid='ignore'):
       iou_c = np.true_divide(inter,union) 
       iou_c[iou_c == np.inf] = 0
       iou_c = np.nan_to_num(iou_c)
    
    #iou = np.sum(iou_c)/n_class_q   
    iou = np.sum(iou_c)/n_class_union  ## Fixed ! 
    
    #if counter.value % 1000 == 0 and counter.value >= 1000:
    #    print('{}'.format(counter.value))
    
    return imgB_id, iou

#%%    
def main(args):
    segment = args.segment
    train_seg = train_uis[(segment-1)*1000: segment*1000]
    
    print('Starting computing iou values for segment ', args.segment)

    tic = time.time()
    iou_dict = defaultdict(dict)
    #for ii in range(2):
    for ii in range(len(train_seg)):    
        anchor = train_seg[ii]
        #counter = Counter()
        p = Pool(30)
        func = partial(compute_iou, anchor)
        results = p.map(func, train_uis)
        temp_ids, temp_ious = map(list, zip(*results))
        
        #sort it [may be optional, was done for selecting top 200 images]
        temp_ids_s =  [y for _,y in sorted(zip(temp_ious,temp_ids), reverse =True)]
        temp_ious_s = [x for x,_ in sorted(zip(temp_ious,temp_ids), reverse =True)]
        
        iou_dict[anchor]['ids_g40'] = [x for x,y  in zip(temp_ids_s,temp_ious_s) if y>0.4]
        iou_dict[anchor]['ious_g40'] = [y for x,y  in zip(temp_ids_s,temp_ious_s) if y>0.4]
        
        ids_b2040 = [x for x,y  in zip(temp_ids_s,temp_ious_s) if y>0.2 and y<0.4 ]
        n_b2040 = min (len(ids_b2040), 200)
        iou_dict[anchor]['ids_b40'] = random.sample(ids_b2040, n_b2040)
        
        if ii%1000 == 0:
            print("Done", ii)    
            toc = time.time() - tic
            print('Elapsed time: ', toc/3600, ' hrs')
        p.close()
    
    save_filename = 'iouValues_segment1000_%s.pkl'%(args.segment)
    with open(save_filename, 'wb') as f:
        pickle.dump(iou_dict, f)
    print('saved to ', save_filename)    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--segment', default = 6, type=int, metavar='N',  
                        help='segment of the train_uis') 


    args = parser.parse_args()
    main(args)
    