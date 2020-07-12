#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 13:44:17 2019

@author: dipu
"""
import numpy as np 
from PIL import Image
from utils import compute_iou
import time

#%%
def dcg_at_k(r, k, method=1):  
    r = np.asfarray(r)[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))

def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max

#%% IoU 
# Compute the Intersection over Union between the query and retrieved images 
# For each query, iterate over all the bounding boxes one at a time. On each query, get all the elements that belong to the same class as the bbox in the query.
# For all the elements, compute the IoU, and select the best matched element with max Iou, detected element in the query image.  
def get_overall_IOU_ndcg(boundingBoxes,sort_inds,g_fnames,q_fnames):      
    allClasses = boundingBoxes.getClasses()
    classIou = dict([(key, []) for key in allClasses])
    
    aNdcg = np.empty((1,0),float)
    wNdcg =np.empty((1,0),float)
    
    for i in  range((sort_inds.shape[0])):   #Iterate over all the query images 
        
        qImageName = q_fnames[i]
        qBBoxes = boundingBoxes.getBoundingBoxesByImageName(qImageName) 
        
        iouList = []
        weightedIouList = []
        
        time_s = time.time()
        #for j in range(len(g_fnames)):     # Iterate over all the gallery images instead of top-5
        for j in range(5): 
            rImageName = g_fnames[sort_inds[i][j]]
            rBBoxes = boundingBoxes.getBoundingBoxesByImageName(rImageName)
            
            iouTemp = []
            weights = []
            
            #Iterate over each element(boudingbox)
            for bb in qBBoxes:                              # qbbs query bounding boxes
                bb_cordinates = bb.getBoundingBox()              
                bb_class = bb.classId 
                
                #get the bouding box in retrieved image that has same class
                rbbs = [d for d in rBBoxes if d.classId == bb_class]
                
                iouMax = 0
                for rbb in rbbs:
                    assert(rbb.classId == bb_class)
                    rbb_cordinates = rbb.getBoundingBox()
                    iou =  compute_iou(bb_cordinates, rbb_cordinates)
                    if iou > iouMax:
                        iouMax = iou
                
                    if iou <0: 
                        print('Warning!!: Negative iou found ', 'ImageName:', rbb.getImageName(), '  bounding box', rbb.getBoundingBox() )
                    assert(iouMax>=0)    
                
                #Store iou with best matched component label
                iouTemp.append(iouMax)
                weights.append(bb_cordinates[2]*bb_cordinates[3])
                # Update iou into coressponding ClassIou
                classIou[bb_class].append(iouMax)
                
            current_iou = np.mean(iouTemp)     # Average Iou between a query and a retrieved image
            weightTotal = np.sum(weights)
            weights = np.divide(weights, weightTotal)
            current_weightedIou = sum(iouTemp*weights) 
            
            weightedIouList.append(current_weightedIou) 
            iouList.append(current_iou)
        
        aGain = ndcg_at_k(iouList,5)
        wGain = ndcg_at_k(weightedIouList,5)
        
        aNdcg = np.append(aNdcg,aGain)
        wNdcg = np.append(wNdcg,wGain)
        time_e = (time.time() - time_s)/3600
        print('Elasped time for one query: {:.3f}'.format(time_e))
    
    avg_aNdcg = np.mean(aNdcg)
    avg_wNdcg = np.mean(wNdcg)    

    return avg_aNdcg, avg_wNdcg
