#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 10:23:48 2019
utility functions and classes
@author: dipu
"""

import os
import errno
import torch
import os.path as osp
import shutil
import numpy as np
from matplotlib import pyplot as plt
import cv2
import glob
from collections import defaultdict
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

currentPath = os.path.dirname(os.path.realpath(__file__))
# Add lib to PYTHONPATH
libPath = os.path.join(currentPath, '..', '..', 'lib')
add_path(libPath)


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'model_best.pth.tar'))


def load_checkpoint(fpath):
    if osp.isfile(fpath):
        checkpoint = torch.load(fpath)
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))
        
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count         
        
        
def imshow(inp, title=None):
    # Imshow for Tensor.
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def extract_features(data_loader, model):
    model.eval()
    torch.set_grad_enabled(False)
    features = []
    labels = []
   
    for i, (imgs, im_fn) in enumerate(data_loader):
        imgs = imgs.cuda()
        x_enc = model(imgs, training=False)
        outputs = x_enc.detach().cpu().numpy()
        features.append(outputs)
        labels += list(im_fn)
        print(i)
    return features, labels  

def add_bb_into_image(image, bb, color=(255, 0, 0), thickness=2, label=None):
    r = int(color[0])
    g = int(color[1])
    b = int(color[2])

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    fontThickness = 1

    x1, y1, x2, y2 = bb.getBoundingBox()
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)
    cv2.rectangle(image, (x1, y1), (x2, y2), (b, g, r), thickness)
    # Add label
    if label is not None:
        # Get size of the text box
        (tw, th) = cv2.getTextSize(label, font, fontScale, fontThickness)[0]
        # Top-left coord of the textbox
        (xin_bb, yin_bb) = (x1 + thickness, y1 - th + int(12.5 * fontScale))
        # Checking position of the text top-left (outside or inside the bb)
        if yin_bb - th <= 0:  # if outside the image
            yin_bb = y1 + th  # put it inside the bb
        r_Xin = x1 - int(thickness / 2)
        r_Yin = y1 - th - int(thickness / 2)
        # Draw filled rectangle to put the text in it
        cv2.rectangle(image, (r_Xin, r_Yin - thickness),
                      (r_Xin + tw + thickness * 3, r_Yin + th + int(12.5 * fontScale)), (b, g, r),
                      -1)
        cv2.putText(image, label, (xin_bb, yin_bb), font, fontScale, (0, 0, 0), fontThickness,
                    cv2.LINE_AA)
    return image

def compute_iou(boxA, boxB):
    #convert boxes to X1Y1X2Y2 format
    boxA = [boxA[0], boxA[1], boxA[0]+boxA[2], boxA[1]+boxA[3]]
    boxB = [boxB[0], boxB[1], boxB[0]+boxB[2], boxB[1]+boxB[3]]
    
    if boxA[0] > boxB[2]:
        return 0  # boxA is right of boxB
    if boxB[0] > boxA[2]:
        return 0  # boxA is left of boxB
    if boxA[3] < boxB[1]:
        return 0  # boxA is above boxB
    if boxA[1] > boxB[3]:
        return 0  # boxA is below boxB
   
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
#    interArea = (xB - xA + 1) * (yB - yA + 1)
#    boxAArea =  (boxA[2] - boxA[0] +1) * (boxA[3] - boxA[1]+1)
#    boxBArea =  (boxB[2] - boxB[0] +1) * (boxB[3] - boxB[1]+1)
    
    interArea = (xB - xA ) * (yB - yA)
    boxAArea =  (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea =  (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    if iou == np.nan:
        print('Something Wrong!, nan Iou values!!!')
        print('Pause')
    return iou
    
    
    
#%% Ploting retrievals
def plot_retrieved_images_and_uis(sort_inds,q_fnames, g_fnames, model_name, \
                                  avgIouArray, weightedIouArray, \
                                  avgClassIouArray, weightedClassIouArray, \
                                  avgPixAccArray, weightedPixAccArray):
    from PIL import Image
    
    base_im_path = '/mnt/scratch/Dipu/RICO/combined/'
    base_ui_path = '/mnt/scratch/Dipu/RICO/semantic_annotations/'
    
    for i in range((sort_inds.shape[0])): #range(1): 
#    for i in  range(2):    
        q_path = base_im_path + q_fnames[i] + '.jpg'
        q_img  =  Image.open(q_path).convert('RGB')
        q_ui_path = base_ui_path + q_fnames[i] + '.png'
        q_ui = Image.open(q_ui_path).convert('RGB')
        
        fig, ax = plt.subplots(2,6, figsize=(30, 12), constrained_layout=True)
        plt.setp(ax,  xticklabels=[],  yticklabels=[])
        fig.suptitle('Query-%s, %s (Gallery_Only-Set)'%(i, model_name), fontsize=20)
        fig = plt.figure(1)
#        fig.set_size_inches(30, 12)
#        plt.subplots_adjust(bottom = 0.1, top=10)
        #f1 = fig.add_subplot(2,6,1)
        
        ax[0,0].imshow(q_ui)
        ax[0,0].axis('off')
        ax[0,0].set_title('Query: %s '%(i) + q_fnames[i] + '.png')
        ax[1,0].imshow(q_img)
        ax[1,0].axis('off') 
        ax[1,0].set_title('Query: %s '%(i) + q_fnames[i] + '.jpg')
        #plt.pause(0.1)
     
        for j in range(5):
            path = base_im_path + g_fnames[sort_inds[i][j]] + '.jpg'
           # print(g_fnames[sort_inds[i][j]] )
            im = Image.open(path).convert('RGB')
            ui_path = base_ui_path + g_fnames[sort_inds[i][j]] + '.png'
            #print(g_fnames[sort_inds[i][j]]) 
            ui = Image.open(ui_path).convert('RGB')
            
            ax[0,j+1].imshow(ui)
            ax[0,j+1].axis('off')
            ax[0,j+1].set_title('Rank: %s  '%(j+1)  + g_fnames[sort_inds[i][j]] \
              + '.png\nAvg IoU: %.3f'%(avgIouArray[i][j])+  '\nWeighted IoU: %.3f'%(weightedIouArray[i][j])\
              + '\nAvg Classwise IoU: %.3f'%(avgClassIouArray[i][j])+  '\nWeighted Classwise IoU: %.3f'%(weightedClassIouArray[i][j]) \
              + '\nAvg PixAcc: %.3f'%(avgPixAccArray[i][j])+  '\nWeighted PixAcc: %.3f'%(weightedPixAccArray[i][j])) 
    
            
            ax[1,j+1].imshow(im)
            ax[1,j+1].axis('off')
            ax[1,j+1].set_title('Rank: %s  '%(j+1) + g_fnames[sort_inds[i][j]] + '.jpg')
            
#        directory =  'Retrieval_Results_Iou_PixAcc/{}/Gallery_Only/'.format(model_name)
        directory =  'Retrieval_Results_Iou_Class_iou_PixAcc/{}/Gallery_Only/'.format(model_name)
        if not os.path.exists(directory):
            os.makedirs(directory)  
            
        plt.savefig( directory + str(i) + '.png')
       # plt.pause(0.1)
        plt.close()
        #print('Wait')
        print('Plotting the retrieved images: {}'.format(i))

#plot_retrieved_images_and_uis(sort_inds,q_fnames, g_fnames, model_name, avgIouArray, weightedIouArray, avgClassIouArray, weightedClassIouArray, avgPixAccArray, weightedPixAccArray)

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

       
def getBoundingBoxes(data_dir = '/mnt/scratch/Dipu/RICO/semantic_annotations/'):
    allBoundingBoxes = BoundingBoxes()
    
    files = glob.glob(data_dir+ "*.json")
    for file in files:
        imageName = os.path.split(file)[1]
        imageName = imageName.replace(".json", "")
        print(imageName)
        
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
            
    return allBoundingBoxes



#%%
#test dataset creation    
#boundingBoxes = getBoundingBoxes()    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    