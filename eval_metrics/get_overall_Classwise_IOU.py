#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 10:44:15 2019

@author: dipu
"""
import numpy as np
from PIL import Image
import time

#%% IoU Classwise

def get_overall_Classwise_IOU(boundingBoxes,sort_inds,g_fnames,q_fnames, topk = [1,5,10,20,40]):
    #data_dir = '/mnt/amber/scratch/Dipu/RICO/semantic_annotations/'
    n_topk = max(topk)
    n_query = len(q_fnames)
    
    avgClassIouArray = np.zeros((n_query,n_topk))
    weightedClassIouArray = np.zeros((n_query,n_topk))
    allClasses = boundingBoxes.getClasses()
    classwiseClassIou = dict([(key, []) for key in allClasses])
    
    for i in  range((sort_inds.shape[0])):   #Iterate over all the query images 
        ts = time.time()
        qImageName = q_fnames[i]
        #q_img  =  Image.open(data_dir+qImageName+'.png').convert('RGB')
        #q_img_size = q_img.size 
        q_img_size = (1440, 2560)
        qBBoxes = boundingBoxes.getBoundingBoxesByImageName(qImageName) 
        qClasses = list(set([d.classId for d in qBBoxes]))
        
        for j in range(n_topk):     # Iterate over top-5 retrieved images
            rImageName = g_fnames[sort_inds[i][j]]
            rBBoxes = boundingBoxes.getBoundingBoxesByImageName(rImageName)
            
            iouTemp = []
            weights = []
            
            #Iterate over each element(boudingbox)
            for c in qClasses:                               # qbbs query bounding boxes
                mask1 = np.zeros(q_img_size, dtype=np.uint8) 
                mask2 = np.zeros(q_img_size).astype(np.uint8) 
                
                c_qboxes = [b for b in qBBoxes if b.classId == c]
                c_rboxes = [b for b in rBBoxes if b.classId == c]
            
                for cqbox in c_qboxes:
                    bb = cqbox.getBoundingBox()
                    mask1[bb[0] : bb[0]+bb[2], bb[1] : bb[1]+bb[3]] = 1
    #            ax[0,1].imshow(Image.fromarray(np.transpose(mask1)))
                
                for crbox in c_rboxes:
                    bb = crbox.getBoundingBox()
                    mask2[bb[0]: bb[0]+ bb[2], bb[1] : bb[1]+bb[3]] = 1
                
                intersec = np.sum(np.logical_and(mask1, mask2))
                union = np.sum(np.logical_or(mask1, mask2))
                iou_c = intersec/union
                
                iouTemp.append(iou_c)
                weights.append(np.sum(mask1))
    
                # Accumuate the pixel acc for all classes    
                classwiseClassIou[c].append(iou_c)  
            
            avgClassIouArray[i][j] = np.mean(iouTemp)
            
            weightTotal = np.sum(weights)
            weightvalues = np.divide(weights, weightTotal)
            weightedClassIouArray[i][j] = np.sum(iouTemp*weightvalues)
        
        #print('Time for query{} = {}'.format(i, time.time()-ts))
        #print('Computing Classwise IoU: {}/{} in time {}'.format(i,n_query, time.time()-ts))     
        #ts = time.time()
    
    overallMeanClassIou_list = []
    overallMeanWeightedClassIou_list = []
    

    
    for k in topk:    
        meanAvgPixAcc = np.mean(avgClassIouArray[:,:k], axis=1)
        overallMeanClassIou = np.mean(meanAvgPixAcc)
        overallMeanClassIou_list.append(overallMeanClassIou)
        
        meanWeightedPixAcc = np.mean(weightedClassIouArray[:,:k], axis=1)
        overallMeanWeightedClassIou = np.mean(meanWeightedPixAcc)
        overallMeanWeightedClassIou_list.append(overallMeanWeightedClassIou)
    print('Completed computing Classwise IoU: {}/{}'.format(i+1,n_query))
    
    return overallMeanClassIou_list, overallMeanWeightedClassIou_list, avgClassIouArray  
      

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
