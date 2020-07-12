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
import pickle
from scipy.spatial.distance import cdist
import numpy as np
import init_paths

from test_dataloader import *
from test_dataloader import RICO_ComponentDataset
#from model import GraphEncoderRasterDecoder
#from models.model import GraphEncoderRasterDecoder
import models
import opts_dml
import os 
import glob
import json
from collections import defaultdict

from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes

import torch.nn.functional as F

from utils import mkdir_if_missing, load_checkpoint
from eval_metrics.get_overall_IOU import get_overall_IOU
from eval_metrics.get_overall_Classwise_IOU import get_overall_Classwise_IOU
from eval_metrics.get_overall_pix_acc import get_overall_pix_acc
from eval_metrics.get_overall_ClasswiseIou_ndcg import get_overall_ClasswiseIou_ndcg
from eval_metrics.get_overall_PixAcc_ndcg import get_overall_PixAcc_ndcg 

#data_dir = '/mnt/scratch/Dipu/RICO/semantic_annotations/'
data_dir = '/mnt/amber/scratch/Dipu/RICO/semantic_annotations/'

#def add_path(path):
#    if path not in sys.path:
#        sys.path.insert(0, path)
#

#currentPath = os.path.dirname(os.path.realpath(__file__))
## Add lib to PYTHONPATH
#libPath = os.path.join(currentPath, '..', '..', 'lib')
#add_path(libPath)

def main():
    opt = opts_dml.parse_opt()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    onlyGallery = True
    #opt.margin = 0.2
    #opt.lambda_mul = 10.0
    #opt.learning_rate_decay_every = 30
    
    #model_name = 'GraphEncoding_boxF0_25Chann1_ModelUpsamp_SigmFalse_LossMSE_PytorchV1.0'
     
    decoderModel =  'Channel25Decoder' if opt.use_25_images else 'RGBDecoder' 
    #lossFunc = 'MSE_Loss' if opt.loss == 'mse' else 'BCE' if opt.loss == 'bce' else  'Huber' if opt.loss== 'huber' else 'L1_Loss'
    lossFunc = 'MSE_Loss' if opt.loss == 'mse' else 'BCE' if opt.loss == 'bce' else  'Huber' if opt.loss== 'huber' else 'L1_loss' if opt.loss=='l1' else 'wMSE_Loss2' if opt.loss == 'wmse' else ''
    deconvModel = 'Upsample' if opt.decoder_model == 'upsample' or opt.decoder_model == 'upsampleRGB' or opt.decoder_model == 'upsample_dim2688' else 'Strided'

    if opt.use_directed_graph:
        model_path = '/home/dipu/codes/GraphEncoding-RICO/trained_models_GCN_DML/GraphEncoder_Models_48K_samples_DirectedGraph/'+decoderModel+ '/' +deconvModel + '/' + lossFunc + '/'
    else:
        model_path = '/home/dipu/codes/GraphEncoding-RICO/trained_models_GCN_DML/GraphEncoder_Models_48K_samples/'+decoderModel+ '/' +deconvModel + '/' + lossFunc + '/'
    #model_name = 'GraphEncoding_boxF0_25Chann1_ModelUpsamp_SigmFalse_LossMSE_PytorchV1.0'

    model_name = 'Model_v2'+ '_box' + str(opt.use_box_feats) + \
                    '_Pretrained' + str(opt.pretrained) +\
                    '_Sigmoid' + str(opt.last_layer_sigmoid) + '_bs' + str(opt.batch_size) + \
                    '_lr' + str(opt.learning_rate) + \
                    '_dr' + str(opt.learning_rate_decay_rate)+ \
                    '_every' + str(opt.learning_rate_decay_every) + \
                    '_gclip' + str(opt.grad_clip) + \
                    '_dimR' + str(opt.dim) + \
                    '_margin' + str(opt.margin) + \
                    '_lmul' + str(opt.lambda_mul)
     
    save_dir = model_path + model_name
    save_file = save_dir + '/result.txt' 

    with open(save_file, 'a') as f:
        for arg in vars(opt):
            f.write('{}: {}\n'.format(arg, getattr(opt,arg)))
                    
    #boundingBoxes = getBoundingBoxes()
    boundingBoxes = getBoundingBoxes_from_info()
    
    #for ep in ['5','10', '15', '20', '25']:
    for ep in [ '25']:
    #for ep in ['10', '30', '35'] :   
        model_file = model_path +  '{}/ckp_ep{}.pth.tar'.format(model_name,ep)
      
        data_transform = transforms.Compose([
                transforms.Resize([255,127]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]) 
        
        #model = GraphEncoderRasterDecoder(opt)
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
    
        #overallMeanIou, overallMeanWeightedIou, classIoU   = get_overall_IOU(boundingBoxes,sort_inds,g_fnames,q_fnames)         
        overallMeanClassIou, overallMeanWeightedClassIou, classwiseClassIoU = get_overall_Classwise_IOU(boundingBoxes,sort_inds,g_fnames,q_fnames, topk = [1,5,10])
        overallMeanAvgPixAcc, overallMeanWeightedPixAcc, classPixAcc = get_overall_pix_acc(boundingBoxes,sort_inds,g_fnames,q_fnames, topk = [1,5,10])     
        
        
        per_query_metrics = {'IoU': classwiseClassIoU, 
                     'PixAcc': classPixAcc}

        pickle.dump(per_query_metrics, open('per_query_metrics_GCN_DML.pkl', "wb"))
                
        
        
#        iou_aDCG, iou_wDCG, iou_avg_aNdcg, iou_avg_wNdcg = get_overall_ClasswiseIou_ndcg(boundingBoxes,sort_inds,g_fnames,q_fnames) 
#        acc_aDCG, acc_wDCG, acc_avg_aNdcg, acc_avg_wNdcg = get_overall_PixAcc_ndcg(boundingBoxes,sort_inds,g_fnames,q_fnames) 

        print('\n\nep:%s'%(ep))
        print(model_name)
        print('GAlleryOnly Flag:', onlyGallery)
        #print('The overallMeanIou = {:.3f}  '.format(overallMeanIou))
        #print('The overallMeanWeightedIou = {:.3f}'.format(overallMeanWeightedIou))
        
#        print('The overallMeanClassIou = {:.3f}'.format(overallMeanClassIou))
#        print('The overallMeanWeightedClassIou = {:.3f}'.format(overallMeanWeightedClassIou))
#        print('The overallMeanAvgPixAcc = {:.3f}'.format(overallMeanAvgPixAcc))
#        print('The overallMeanWeightedPixAcc = {:.3f} '.format(overallMeanWeightedPixAcc))
        
        print('The overallMeanClassIou =  ' + str([ '{:.3f}'.format(x) for x in overallMeanClassIou]) + '\n')        
        print('The overallMeanWeightedClassIou =  ' + str([ '{:.3f}'.format(x) for x in overallMeanWeightedClassIou]) + '\n')
        print('The overallMeanAvgPixAcc =  ' + str([ '{:.3f}'.format(x) for x in overallMeanAvgPixAcc]) + '\n')
        print('The overallMeanWeightedPixAcc =  ' + str([ '{:.3f}'.format(x) for x in overallMeanWeightedPixAcc]) + '\n')
        
#        print('The iou_aDCG =  ' + str([ '{:.3f}'.format(x) for x in iou_aDCG]) + '\n')
#        print('The iou_wDCG =  ' + str([ '{:.3f}'.format(x) for x in iou_wDCG]) + '\n')
#        print('The iou_avg_aNdcg =  ' + str([ '{:.3f}'.format(x) for x in iou_avg_aNdcg]) + '\n')
#        print('The iou_avg_wNdcg =  ' + str([ '{:.3f}'.format(x) for x in iou_avg_wNdcg]) + '\n')
#        print('The acc_aDCG =  ' + str([ '{:.3f}'.format(x) for x in acc_aDCG]) + '\n')
#        print('The acc_wDCG =  ' + str([ '{:.3f}'.format(x) for x in acc_wDCG]) + '\n')
#        print('The acc_avg_aNdcg =  ' + str([ '{:.3f}'.format(x) for x in acc_avg_aNdcg]) + '\n')
#       

        
        #Save results
    #    save_dir = 'Results/%s/'%(model_name)
    #    mkdir_if_missing(save_dir)
    #    savefile = save_dir + 'results.p'
    #    results = {'overallMeanIou': overallMeanIou, 'overallMeanWeightedIou': overallMeanWeightedIou, 'classIoU': classIoU, \
    #                'overallMeanClassIou': overallMeanClassIou, 'overallMeanWeightedClassIou': overallMeanWeightedClassIou,  'classwiseClassIoU': classwiseClassIoU, \
    #                'overallMeanAvgPixAcc': overallMeanAvgPixAcc, 'overallMeanWeightedPixAcc': overallMeanWeightedPixAcc, 'classPixAcc':classPixAcc \
    #                }
    #    
    #    pickle.dump(results, open(savefile, "wb"))
        
        

        with open(save_file, 'a') as f:
                f.write('\n\ep: {}\n'.format(ep))
                f.write('Model name: {}\n'.format(model_name))
                f.write('GAlleryOnly Flag: {}\n'.format(onlyGallery))
                #f.write('The overallMeanIou = {:.3f}\n'.format(overallMeanIou))
                #f.write('The overallMeanWeightedIou = {:.3f}\n'.format(overallMeanWeightedIou))
                
                f.write('The overallMeanClassIou =  ' + str([ '{:.3f}'.format(x) for x in overallMeanClassIou]) + '\n')
                f.write('The overallMeanWeightedClassIou =  ' + str([ '{:.3f}'.format(x) for x in overallMeanWeightedClassIou]) + '\n')
                f.write('The overallMeanAvgPixAcc =  ' + str([ '{:.3f}'.format(x) for x in overallMeanAvgPixAcc]) + '\n')
                f.write('The overallMeanWeightedPixAcc =  ' + str([ '{:.3f}'.format(x) for x in overallMeanWeightedPixAcc]) + '\n')
            
                
                
#                f.write('The overallMeanClassIou = {:.3f})\n'.format(overallMeanClassIou))
#                f.write('The overallMeanWeightedClassIou = {:.3f}\n'.format(overallMeanWeightedClassIou))
#                f.write('The overallMeanAvgPixAcc = {:.3f}\n'.format(overallMeanAvgPixAcc))
#                f.write('The overallMeanWeightedPixAcc = {:.3f}\n'.format(overallMeanWeightedPixAcc))
#                
#                f.write('iou_aDCG = {:.3f} \n '.format(iou_aDCG))
#                f.write('iou_wDCG = {:.3f} \n'.format(iou_wDCG))
#                f.write('iou_avg_aNdcg = {:.3f}\n'.format(iou_avg_aNdcg))
#                f.write('iou_avg_wNdcg = {:.3f}\n'.format(iou_avg_wNdcg))
#        
#                f.write('acc_aDCG = {:.3f}\n  '.format(acc_aDCG))
#                f.write('acc_wDCG = {:.3f}\n'.format(acc_wDCG))
#                f.write('acc_avg_aNdcg = {:.3f}\n'.format(acc_avg_aNdcg))
#                f.write('acc_avg_wNdcg = {:.3f}\n'.format(acc_avg_wNdcg))
                
#        fig_save_dir = '%s/Results/'%(save_dir)
#        mkdir_if_missing(fig_save_dir)
#        plot_classwiseResults(classIoU, fig_save_dir , 'classIoU')
#        plot_classwiseResults(classwiseClassIoU, fig_save_dir, 'classwiseClassIoU')
#        plot_classwiseResults(classPixAcc, fig_save_dir, 'classPixAcc')
   
    
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
        
    
#%% Preparing the dataset
def parse_ui_elements(sui):
    """
    Parse the json file iteratively using recursion,, un winding all the nested chilfre
    returns the dictionay of elements   
    """
    global counter
    counter = 0
    elements = defaultdict(dict)
    
    def recurse(sui):
        global counter
        n_uis = len(sui['children'])
        for i in range(n_uis):                
            [x1, y1, x2, y2] = sui['children'][i]['bounds']
            elements[counter]['component_Label'] = sui['children'][i]['componentLabel']
            elements[counter]['x'] = x1
            elements[counter]['y'] = y1
            elements[counter]['w'] = x2-x1
            elements[counter]['h'] = y2-y1
            elements[counter]['iconClass'] = sui['children'][i].get('iconClass') 
            elements[counter]['textButtonClass'] = sui['children'][i].get('textButtonClass')
            counter +=1
            if sui['children'][i].get('children') != None:
                recurse(sui['children'][i])
    recurse(sui)        
    return elements, counter 

       
def getBoundingBoxes(data_dir = '/mnt/amber/scratch/Dipu/RICO/semantic_annotations/'):
    allBoundingBoxes = BoundingBoxes()
    
    files = glob.glob(data_dir+ "*.json")
    for file in files:
        imageName = os.path.split(file)[1]
        imageName = imageName.replace(".json", "")
        
        with open(file, "r") as f:
           sui = json.load(f)   # sui = semantic ui annotation.
           
        elements, count = parse_ui_elements(sui)
        for i in range(count):
            box = elements[i]
            bb = BoundingBox(
                imageName,
                box['component_Label'],
                box['x'],
                box['y'],
                box['w'],
                box['h'],
                iconClass=box['iconClass'],
                textButtonClass=box['textButtonClass'])
            allBoundingBoxes.addBoundingBox(bb) 
    print('Collected {} bounding boxes from {} images'. format(allBoundingBoxes.count(), len(files) ))         
#    testBoundingBoxes(allBoundingBoxes)
    return allBoundingBoxes


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

def testBoundingBoxes(samples = ['5', '999']):
    #samples = ['28970', '62918']
    #Visualize if every colored element is plotted or not.
    #boundingBoxes = getBoundingBoxes()
    boundingBoxes = getBoundingBoxes_from_info()
    from matplotlib import pyplot as plt
    from PIL import Image
    import matplotlib.patches as patches
#    samples = ['28970', '62918']
    
    base_ui_path = '/mnt/amber/scratch/Dipu/RICO/semantic_annotations/'
    base_im_path = '/mnt/amber/scratch/Dipu/RICO/combined/'
    
    for sample in samples:
        img = base_ui_path + sample + '.png'
        img = Image.open(img).convert('RGB')
        img2 = base_im_path + sample + '.jpg'
        img2 = Image.open(img2).convert('RGB')
        
        fig, ax = plt.subplots(1,2)
        plt.setp(ax,  xticklabels=[], yticklabels=[])
        ax[0].imshow(img2)
        ax[1].imshow(img)
        bbs = boundingBoxes.getBoundingBoxesByImageName(sample)
        for bb in bbs:
            bb_cordinates = bb.getBoundingBox()
            bb_class = bb.classId
#            if bb_cordinates[2] < 0:
            rect = patches.Rectangle((bb_cordinates[0], bb_cordinates[1]), bb_cordinates[2], bb_cordinates[3], linewidth=2, edgecolor='r', facecolor= 'none')
            ax[1].add_patch(rect)
            ax[1].text(bb_cordinates[0], bb_cordinates[1],  bb_class,  fontsize=8, color= 'r', verticalalignment='top')
        plt.show()    
      



#%%
if __name__ == '__main__':
    main()
