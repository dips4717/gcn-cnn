#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 11:41:06 2019
Training code for GraphEmbedding for RICO dataset
@author: dipu
"""

import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import init_paths

from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes

from dataloader import *
from dataloader import RICO_ComponentDataset
#from model_25_channel_out import GraphEncoderRasterDecoder
#from models.model import GraphEncoderRasterDecoder
import models

#from dataloader import *
#from dataloader import RICO_ComponentDataset
#from model import GraphEncoderRasterDecoder
import opts_dml
import shutil
import os
import os.path as osp
import errno
import time 
import numpy as np
from scipy.spatial.distance import cdist


from eval_metrics.get_overall_Classwise_IOU import get_overall_Classwise_IOU
from eval_metrics.get_overall_pix_acc import get_overall_pix_acc
from evaluate import getBoundingBoxes_from_info
#%%
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

def save_checkpoint(state, is_best, fpath='checkpoint.pth'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'model_best.pth'))

def load_checkpoint(fpath):
    if osp.isfile(fpath):
        checkpoint = torch.load(fpath)
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))
        
def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise 


def set_lr2(optimizer, decay_factor):
    for group in optimizer.param_groups:
        group['lr'] = group['lr'] * decay_factor
    print('\n', optimizer, '\n')            
        
def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)        


#%%
def main(opt):
    print(opt)
    data_transform = transforms.Compose([ # Only used if decoder is trained using 3-Channel RBG (not 25Channel Images)
            transforms.Resize([254,126]),  
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    
    save_dir = 'trained_model/model_dec_{}_dim{}'.format(opt.decoder_model, opt.dim)
    mkdir_if_missing(save_dir)
    print ('Output dir: ', save_dir)
    
    loader = RICO_ComponentDataset(opt, data_transform) 
    model = models.create(opt.decoder_model, opt)
    model = model.cuda() 
   
    print(model)
    
    if opt.loss =='mse':   # MSE Loss works the best
        criterion = nn.MSELoss()
    elif opt.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([30.0]).cuda())
    elif opt.loss == 'huber':
        criterion = nn.SmoothL1Loss()
    elif opt.loss == 'l1':
        criterion = nn.L1Loss()   
           
    optimizer = torch.optim.Adam(model.parameters(), opt.learning_rate)  
    
    epoch_done = True
    iteration = 0
    epoch = 0
    
    model.train()
    torch.set_grad_enabled(True)
    time_s = time.time()
    
    boundingBoxes = getBoundingBoxes_from_info()
    
    while True:
        if epoch_done:
            if epoch > opt.learning_rate_decay_start+1 and epoch%opt.learning_rate_decay_every==0 and opt.learning_rate_decay_start >= 0  :
                decay_factor = opt.learning_rate_decay_rate
                set_lr2(optimizer, decay_factor)     
                
            losses = AverageMeter()
            epoch_done = False
        
        # Load a batch of data from train split
        data =  loader.get_batch('train')
        images = data['images'].cuda()
        sg_data = {key: torch.from_numpy(data['sg_data'][key]).cuda() for key in data['sg_data']}
        
        #2. Forward model and compute loss
#        torch.cuda.synchronize()   # Waits for all kernels in all streams on a CUDA device to complete. 
        optimizer.zero_grad()
        _, out = model(sg_data)
        print ('out size :', out.size() )
        print('images size: ', images.size())
        if opt.decoder_model == 'strided': 
            images = F.interpolate(images, size= [239,111])
        elif opt.decoder_model == 'upsample':
            images = F.interpolate(images, size= [254,126])            
        loss = criterion(out, images) 
        
        # 3. Update model
        loss.backward()
        clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()
        torch.cuda.empty_cache()
        
        losses.update(loss.detach().item())
        
        # Update the iteration and epoch
        iteration += 1
        
        if epoch ==0 and iteration ==1:
            print("Training Started ")
        
        if iteration%1000 == 0:
            elsp_time = (time.time() - time_s)
            print( 'Epoch [%02d] [%05d / %05d] Average_Loss: %.5f' % (epoch+1, iteration*opt.batch_size, len(loader), losses.avg ))
            with open(save_dir+'/log.txt', 'a') as f:
                f.write('Epoch [%02d] [%05d / %05d  ] Average_Loss: %.5f\n'%(epoch+1, iteration*opt.batch_size, len(loader), losses.avg ))
                f.write('Completed {} images in {}'.format(iteration*opt.batch_size, elsp_time))
                
            
            print('Completed {} images in {}'.format(iteration*opt.batch_size, elsp_time))
            time_s = time.time()
            
            #print( 'Epoch [%02d] [%05d ] Average_Loss: %.3f' % (epoch+1, iteration*opt.batch_size, len(loader)))
        
        if data['bounds']['wrapped']:
            epoch += 1
            epoch_done = True 
            iteration = 0 
            
        del data, images, sg_data, out, loss 
    
        if (epoch+1) % 5 == 0 and epoch_done:
            state_dict = model.state_dict()  
            
            save_checkpoint({
            'state_dict': state_dict,
            'epoch': (epoch+1)}, is_best=False, fpath=osp.join(save_dir, 'ckp_ep' + str(epoch + 1) + '.pth.tar'))
            print('Saved the model for epoch {}'.format(epoch+1))
            
            # Also perform the tests
            perform_tests(model, loader, boundingBoxes,  save_dir, epoch)
            
            # set model to training mode again.
            model.train()
            torch.set_grad_enabled(True)
                   
        if epoch > 25:
            break



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
        outputs = x_enc.detach().cpu().numpy()
        feat.append(outputs)
        fnames += [x['id'] for x in data['infos']]
    
        if data['bounds']['wrapped']:
            #print('Extracted features from {} images from {} split'.format(c, split))
            epoch_done = True
    
    print('Extracted features from {} images from {} split'.format(len(fnames), split))
    return feat, fnames


def perform_tests(model, loader, boundingBoxes,  save_dir, ep):
    model.eval()  
    save_file = save_dir + '/result.txt' 
    q_feat, q_fnames = extract_features(model, loader, split='query')
    g_feat, g_fnames = extract_features(model, loader, split='gallery')
    
    onlyGallery = True
    if not(onlyGallery):
        t_feat, t_fnames = extract_features(model, loader, split='train')
        g_feat = np.vstack((g_feat,t_feat))
        g_fnames = g_fnames + t_fnames 

    q_feat = np.concatenate(q_feat)
    g_feat = np.concatenate(g_feat)
    
    distances = cdist(q_feat, g_feat, metric= 'euclidean')
    sort_inds = np.argsort(distances)
      
    overallMeanClassIou, overallMeanWeightedClassIou, classwiseClassIoU = get_overall_Classwise_IOU(boundingBoxes,sort_inds,g_fnames,q_fnames, topk = [1,5,10])
    overallMeanAvgPixAcc, overallMeanWeightedPixAcc, classPixAcc = get_overall_pix_acc(boundingBoxes,sort_inds,g_fnames,q_fnames, topk = [1,5,10])     
    

            
    print('\n\nep:%s'%(ep))
    print(save_dir)
    print('GAlleryOnly Flag:', onlyGallery)
    
    print('overallMeanClassIou =  ' + str([ '{:.3f}'.format(x) for x in overallMeanClassIou]) + '\n')        
    print('overallMeanAvgPixAcc =  ' + str([ '{:.3f}'.format(x) for x in overallMeanAvgPixAcc]) + '\n')
    
           
    with open(save_file, 'a') as f:
        f.write('\n\ep: {}\n'.format(ep))
        f.write('Model name: {}\n'.format(save_dir))
        f.write('GAlleryOnly Flag: {}\n'.format(onlyGallery))     
       
        f.write('overallMeanClassIou =  ' + str([ '{:.3f}'.format(x) for x in overallMeanClassIou]) + '\n')
        f.write('overallMeanAvgPixAcc =  ' + str([ '{:.3f}'.format(x) for x in overallMeanAvgPixAcc]) + '\n')
        
opt = opts_dml.parse_opt()
for arg in vars(opt):
    print(arg, getattr(opt,arg))


os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
if __name__ == '__main__':
    main(opt)

    
    
    