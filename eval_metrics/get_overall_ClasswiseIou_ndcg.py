#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 13:41:42 2019

@author: dipu
"""


import numpy as np
from PIL import Image
data_dir = '/mnt/amber/scratch/Dipu/RICO/semantic_annotations/'

def dcg_at_k(r, k, method=1):  
    r = np.asfarray(r)[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))

def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max

#%% IoU Classwise
    
def get_overall_ClasswiseIou_ndcg(boundingBoxes,sort_inds,g_fnames,q_fnames, topk = [1,5,10,20,40]): 
    n_query = len(q_fnames)
    n_topk = max(topk)
#    aNdcg = np.empty((1,0),float)
#    wNdcg =np.empty((1,0),float)
#    
#    aDCG = np.empty((1,0),float)
#    wDCG =np.empty((1,0),float)
    
    allClasses = boundingBoxes.getClasses()
    classwiseClassIou = dict([(key, []) for key in allClasses])
    
    avgClassIouArray = np.zeros((n_query,n_topk))
    weightedClassIouArray = np.zeros((n_query,n_topk))
    
    for i in  range((sort_inds.shape[0])):   #Iterate over all the query images 
        qImageName = q_fnames[i]
        q_img  =  Image.open(data_dir+qImageName+'.png').convert('RGB')
        qBBoxes = boundingBoxes.getBoundingBoxesByImageName(qImageName) 
        qClasses = list(set([d.classId for d in qBBoxes]))
        
        #iouList = []
        #weightedIouList = []
        
        #for j in range(len(g_fnames)):     # Iterate over top-5 retrieved images
        for j in range(n_topk):
            rImageName = g_fnames[sort_inds[i][j]]
            rBBoxes = boundingBoxes.getBoundingBoxesByImageName(rImageName)
            
            iouTemp = []
            weights = []
            
            #Iterate over each element(boudingbox)
            for c in qClasses:                               # qbbs query bounding boxes
                mask1 = np.zeros(q_img.size, dtype=np.uint8) 
                mask2 = np.zeros(q_img.size).astype(np.uint8) 
                
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
                
                if iou_c <0:
                    print('Warning!!  Negative iou found! iou = ', iou_c, 'rImageName:', rImageName )
                
                iouTemp.append(iou_c)
                weights.append(np.sum(mask1))
    
                # Accumuate the pixel acc for all classes    
                classwiseClassIou[c].append(iou_c) 
            
            current_iou = np.mean(iouTemp)     # Average Iou between a query and a retrieved image
            weightTotal = np.sum(weights)
            weights = np.divide(weights, weightTotal)
            current_weightedIou = sum(iouTemp*weights)     
            
            #weightedIouList.append(current_weightedIou) 
            #iouList.append(current_iou)
            
            avgClassIouArray[i][j] = current_iou
            weightedClassIouArray[i][j] = current_weightedIou
        
        print('Computing NDCG IoU: {}/{}'.format(i,50))
        
        
    aDCG = np.zeros((n_query,n_topk))   # stores avg-DCG for each query for each topk 
    wDCG = np.zeros((n_query,n_topk))
    
    aNdcg = np.zeros((n_query,n_topk))
    wNdcg = np.zeros((n_query,n_topk)) 
    
    for k in range(len(topk)):
        for i in range(n_query):
            dcg_t =  dcg_at_k(avgClassIouArray[i][:], k)
            wdcg_t = dcg_at_k(weightedClassIouArray[i][:], k)
            aDCG[i,k] =  dcg_t
            wDCG[i,k] =  wdcg_t
                          
            aGain = ndcg_at_k(avgClassIouArray[i][:], k)
            wGain = ndcg_at_k(weightedClassIouArray[i][:], k)     
    
            aNdcg[i,k] = aGain
            wNdcg[i,k] = wGain
    
    avg_aDCG = list(np.mean(aDCG,  axis =1))
    avg_wDCG = list(np.mean(wDCG,  axis=1))
    avg_aNdcg = list(np.mean(aNdcg,axis=1))
    avg_wNdcg = list(np.mean(wNdcg,axis=1))    
    
    print('Completed computing NCDG IOU')
    return avg_aDCG, avg_wDCG, avg_aNdcg, avg_wNdcg  