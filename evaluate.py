#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:01:07 2019
Compute the performance metrics for graphencoder model 
performance metrics includes iou,  pixelAccuracy
@author: dipu
"""

import torch
from torchvision import transforms
import torch.nn.functional as F
import pickle
from scipy.spatial.distance import cdist
import numpy as np
import init_paths

from dataloaders.dataloader_test import *
from dataloaders.dataloader_test import RICO_ComponentDataset

import models
import opts_dml
import os 

from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes

from utils import mkdir_if_missing, load_checkpoint
from eval_metrics.get_overall_Classwise_IOU import get_overall_Classwise_IOU
from eval_metrics.get_overall_pix_acc import get_overall_pix_acc



def main():
    opt = opts_dml.parse_opt()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    onlyGallery = True
    opt.use_directed_graph = True
    opt.decoder_model = 'strided'
    opt.dim =1024
    

    boundingBoxes = getBoundingBoxes_from_info()
    #model_file = 'trained_model/model_dec_strided_dim1024_TRI_ep25.pth'
    model_file = 'trained_model/model_dec_strided_dim1024_ep35.pth'
   
      
    data_transform = transforms.Compose([  # Not used for 25Channel_images
            transforms.Resize([255,127]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]) 
    

    model = models.create(opt.decoder_model, opt)
    resume = load_checkpoint(model_file)
    model.load_state_dict(resume['state_dict'])
    model = model.cuda()
    model.eval()
    
    loader = RICO_ComponentDataset(opt, data_transform)     
    
    q_feat, q_fnames = extract_features(model, loader, split='query')
    g_feat, g_fnames = extract_features(model, loader, split='gallery')
    
    
    if not(onlyGallery):
        t_feat, t_fnames = extract_features(model, loader, split='train')
        g_feat = np.vstack((g_feat,t_feat))
        g_fnames = g_fnames + t_fnames 

    q_feat = np.concatenate(q_feat)
    g_feat = np.concatenate(g_feat)
    
    distances = cdist(q_feat, g_feat, metric= 'euclidean')
    sort_inds = np.argsort(distances)

           
    overallMeanClassIou, _, _ = get_overall_Classwise_IOU(boundingBoxes,sort_inds,g_fnames,q_fnames, topk = [1,5,10])
    overallMeanAvgPixAcc, _, _ = get_overall_pix_acc(boundingBoxes,sort_inds,g_fnames,q_fnames, topk = [1,5,10])     
   
    print('The overallMeanClassIou =  ' + str([ '{:.3f}'.format(x) for x in overallMeanClassIou]) + '\n')        
    print('The overallMeanAvgPixAcc =  ' + str([ '{:.3f}'.format(x) for x in overallMeanAvgPixAcc]) + '\n')


    
def extract_features(model, loader, split='gallery'):
    epoch_done = False 
    feat = []
    fnames = [] 
    c=0 
    
    torch.set_grad_enabled(False)
    while epoch_done == False:
        c+=1
        data = loader.get_batch(split)
        sg_data = {key: torch.from_numpy(data['sg_data'][key]).cuda() for key in data['sg_data']}
        x_enc, x_dec = model(sg_data)
        x_enc = F.normalize(x_enc)
        outputs = x_enc.detach().cpu().numpy()
        feat.append(outputs)
        fnames += [x['id'] for x in data['infos']]
    
        if data['bounds']['wrapped']:
            #print('Extracted features from {} images from {} split'.format(c, split))
            epoch_done = True
    
    print('Extracted features from {} images from {} split'.format(len(fnames), split))
    return feat, fnames
        
    

# prepare bounding boxes information for RICO datgsaet
def getBoundingBoxes_from_info(info_file = 'data/rico_box_info.pkl'):
    allBoundingBoxes = BoundingBoxes()
    info = pickle.load(open(info_file, 'rb'))
    #files = glob.glob(data_dir+ "*.json")
    for imageName in info.keys():
        count = info[imageName]['nComponent']
        for i in range(count):
            box = info[imageName]['xywh'][i]
            bb = BoundingBox(
                imageName,
                info[imageName]['componentLabel'][i],
                box[0],
                box[1],
                box[2],
                box[3],
                iconClass=info[imageName]['iconClass'],
                textButtonClass=info[imageName]['textButtonClass'])
            allBoundingBoxes.addBoundingBox(bb) 
    print('Collected {} bounding boxes from {} images'. format(allBoundingBoxes.count(), len(info) ))         
#    testBoundingBoxes(allBoundingBoxes)
    return allBoundingBoxes

  
      



#%%
if __name__ == '__main__':
    main()
