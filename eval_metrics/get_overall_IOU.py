#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Created on Wed Nov 20 10:16:48 2019
Compute the Intersection over Union between the query and retrieved images 
For each query, iterate over all the bounding boxes one at a time. 
On each query, get all the elements that belong to the same class as the bbox in the query.
For all the elements, compute the IoU, and select the best matched element with max Iou, detected element in the query image.  
@author: dipu
"""
import numpy as np
from utils import compute_iou
import time


#%% IoU 
def get_overall_IOU(boundingBoxes,sort_inds,g_fnames,q_fnames): 
    """
        boudningBoxes: boundingBoxes class which is the list all the bounding boxes in the images
        g_fnames = gallyer filenames. without the extension (no .png) eg. '24889' 
    """
    n_query = len(q_fnames)
    avgIouArray = np.zeros((n_query,5))
    weightedIouArray = np.zeros((n_query,5))
    allClasses = boundingBoxes.getClasses()
    classIou = dict([(key, []) for key in allClasses])
    
    for i in  range((sort_inds.shape[0])):   #Iterate over all the query images 
        ts = time.time()
        qImageName = q_fnames[i]
        qBBoxes = boundingBoxes.getBoundingBoxesByImageName(qImageName) 
        
        for j in range(5):     # Iterate over top-5 retrieved images
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
                
                iouMax = 0 # sys.float_info.min
                for rbb in rbbs:
                    assert(rbb.classId == bb_class)
                    rbb_cordinates = rbb.getBoundingBox()
                    iou =  compute_iou(bb_cordinates, rbb_cordinates)
                    if iou > iouMax:
                        iouMax = iou
                    if iou <0: 
                        print('Warning!!: Negative iou found ', 'ImageName:', rbb.getImageName(), '  bounding box', rbb.getBoundingBox())
                    assert(iouMax>=0)        
                #Store iou with best matched component label
                iouTemp.append(iouMax)
                weights.append(bb_cordinates[2]*bb_cordinates[3])
                # Update iou into coressponding ClassIou
                classIou[bb_class].append(iouMax)
                
            avgIouArray[i][j] = np.mean(iouTemp)     # Average Iou between a query and a retrieved image
            
            weightTotal = np.sum(weights)
            weights = np.divide(weights, weightTotal)
            weightedIou = sum(iouTemp*weights) 
            weightedIouArray[i][j] = weightedIou
        
        #print('Computing IoU metric: {}/{}'.format(i,50))
        #print('Time for query{} = {}'.format(i, time.time()-ts))
        print('Computing  IoU: {}/{}  in time {}'.format(i,n_query, time.time()-ts))     
        ts = time.time()
        
    meanAvgIou = np.mean(avgIouArray, axis=1)
    overallMeanIou = np.mean(meanAvgIou)
    
    meanWeightedIou = np.mean(weightedIouArray, axis=1)    
    overallMeanWeightedIou = np.mean(meanWeightedIou)  
    
    print('Completed computing IoU metric: {}/{}'.format(i+1,n_query))

    return overallMeanIou, overallMeanWeightedIou, classIou      

