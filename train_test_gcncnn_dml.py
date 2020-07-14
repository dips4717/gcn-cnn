#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:41:48 2020
Training code for GCN-CNN autoencoder with triplet network for metric learning.
Dataloader loads triplets of graph data, images.. 
Network is trained with reconstruction loss and triplet ranking loss.

@author: dipu
"""


import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import init_paths

from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes

from dataloader_triplet import *
from dataloader_triplet import RICO_TripletDataset
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
from perform_tests_2 import getBoundingBoxes

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
        
def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise 

#def set_lr(optimizer, lr):
#    for group in optimizer.param_groups:
#        group['lr'] = lr
        
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
def weighted_mse_loss(input, target, weight):
    mse = (input - target) ** 2      # B * C * W * H
    mse = torch.mean(mse, dim =(2,3))  #  B*C
    return torch.mean(weight*mse)
             
def compute_MSE_weights(images):
    weight = torch.sum(images,dim=(2,3))
    sumw = torch.sum(weight, dim=1, keepdim=True)
    weight = weight/sumw
    weight = 1/weight 
    weight[weight==float('inf')] = 0
    weight = weight / weight.sum(1, keepdim=True)
    weight = 10*weight
    weight[weight==0] = 1
    return weight

def weighted_mse_loss_2(input, target, weights):
    mse = (input - target) ** 2      # B * C * W * H
    mse = torch.mean(mse, dim =(0,2,3))  #  C
    return torch.mean(weights*mse) 


class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.d = nn.PairwiseDistance(p=2)
        print ('Triplet loss intilized with margin {}'.format(self.margin))
    
    def forward(self, anchor, positive, negative, size_average=True):
        distance = self.d(anchor, positive) - self.d(anchor, negative) + self.margin
        loss = torch.mean(torch.max(distance, torch.zeros_like(distance)))
        return loss
#%%
def main(opt):
    print(opt)
    data_transform = transforms.Compose([
            transforms.Resize([254,126]),  # transforms.Resize([254,126])
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    decoderModel =  'Channel25Decoder' if opt.use_25_images else 'RGBDecoder' 
    #lossFunc = 'MSE_Loss' if opt.loss == 'mse' else 'BCE' if opt.loss == 'bce' else  'Huber' if opt.loss== 'huber' else 'L1_Loss'
    lossFunc = 'MSE_Loss' if opt.loss == 'mse' else 'BCE' if opt.loss == 'bce' else  'Huber' if opt.loss== 'huber' else 'L1_loss' if opt.loss=='l1' else 'wMSE_Loss2' if opt.loss == 'wmse' else ''
    deconvModel = 'Upsample' if opt.decoder_model == 'upsample' or opt.decoder_model == 'upsampleRGB' or opt.decoder_model == 'upsample_dim2688' else 'Strided'
    
    if opt.use_directed_graph:
        save_dir = '/home/dipu/codes/GraphEncoding-RICO/trained_models_GCN_DML/GraphEncoder_Models_48K_samples_DirectedGraph/'+decoderModel+ '/' +deconvModel + '/' + lossFunc + '/'
    else:
        save_dir = '/home/dipu/codes/GraphEncoding-RICO/trained_models_GCN_DML/GraphEncoder_Models_48K_samples/'+decoderModel+ '/' +deconvModel + '/' + lossFunc + '/'
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
                    '_lmul' + str(opt.lambda_mul) + \
                    '_hardM' + str(opt.hardmining) +\
                    '_7Dfeat' + str(opt.use_7D_feat) #+\
                    #'_readout' + str(opt.readout) +\
                    #'_xyfeatMod' + str(opt.xy_modified_feat) +\
                    #'_contFeat'+ str(opt.containment_feat)  
   
    save_dir = save_dir + model_name
    mkdir_if_missing(save_dir)
    
    loader = RICO_TripletDataset(opt, data_transform) 
    loader_test = RICO_ComponentDataset(opt, data_transform)
    model = models.create(opt.decoder_model, opt)
    model = model.cuda()
    #model = GraphEncoderRasterDecoder(opt)
    
    if opt.pretrained:
        if opt.decoder_model == 'upsample' and opt.use_directed_graph == False:
            if opt.dim == 512:
                model_path = 'trained_models/GraphEncoder_Models/Channel25Decoder/Upsample/MSE_Loss/Model_v2_boxTrue_PretrainedFalse_SigmoidFalse_bs20_lr0.001_dr0.1_every10_dim512/'
            elif opt.dim == 128:
                model_path = 'trained_models/GraphEncoder_Models/Channel25Decoder/Upsample/MSE_Loss/Model_v2_boxTrue_PretrainedFalse_SigmoidFalse_bs20_lr0.001_dr0.1_every10_dim128/'
            elif opt.dim == 256:
                model_path = 'trained_models/GraphEncoder_Models/Channel25Decoder/Upsample/MSE_Loss/Model_v2_boxTrue_PretrainedFalse_SigmoidFalse_bs20_lr0.001_dr0.1_every10_dim256/'
            elif opt.dim == 1024:
                model_path = 'trained_models/GraphEncoder_Models/Channel25Decoder/Upsample/MSE_Loss/Model_v2_boxTrue_PretrainedFalse_SigmoidFalse_bs20_lr0.001_dr0.1_every10_dim1024/'
            elif opt.dim == 2048:
                model_path = 'trained_models/GraphEncoder_Models/Channel25Decoder/Upsample/MSE_Loss/Model_v2_boxTrue_PretrainedFalse_SigmoidFalse_bs20_lr0.001_dr0.1_every10_dim2048/'
        
        elif opt.decoder_model == 'upsample' and opt.use_directed_graph == True:  
            if opt.dim == 128:
                model_path = 'trained_models/GraphEncoder_Models_DirectedGraph/Channel25Decoder/Upsample/MSE_Loss/Model_v2_boxTrue_PretrainedFalse_SigmoidFalse_bs12_lr0.001_dr0.1_every10_dim128/'
            elif opt.dim == 256:
                model_path = 'trained_models/GraphEncoder_Models_DirectedGraph/Channel25Decoder/Upsample/MSE_Loss/Model_v2_boxTrue_PretrainedFalse_SigmoidFalse_bs12_lr0.001_dr0.1_every10_dim256/'    
            elif opt.dim == 512:
                model_path = 'trained_models/GraphEncoder_Models_DirectedGraph/Channel25Decoder/Upsample/MSE_Loss/Model_v2_boxTrue_PretrainedFalse_SigmoidFalse_bs12_lr0.001_dr0.1_every10_dim512/'
            elif opt.dim == 1024:
                model_path = 'trained_models/GraphEncoder_Models_DirectedGraph/Channel25Decoder/Upsample/MSE_Loss/Model_v2_boxTrue_PretrainedFalse_SigmoidFalse_bs12_lr0.001_dr0.1_every10_dim1024/'
            elif opt.dim ==  2048:
                model_path = 'trained_models/GraphEncoder_Models_DirectedGraph/Channel25Decoder/Upsample/MSE_Loss/Model_v2_boxTrue_PretrainedFalse_SigmoidFalse_bs12_lr0.001_dr0.1_every10_dim2048/'
            
        elif opt.decoder_model == 'strided' and opt.use_directed_graph == False:
            if opt.dim == 128:
                model_path = 'trained_models/GraphEncoder_Models/Channel25Decoder/Strided/MSE_Loss/Model_v2_boxTrue_PretrainedFalse_SigmoidFalse_bs20_lr0.001_dr0.1_every10_dim128/'
            elif opt.dim == 256:
                model_path = 'trained_models/GraphEncoder_Models/Channel25Decoder/Strided/MSE_Loss/Model_v2_boxTrue_PretrainedFalse_SigmoidFalse_bs20_lr0.001_dr0.1_every10_dim256/'
            elif opt.dim == 512:
                model_path = 'trained_models/GraphEncoder_Models/Channel25Decoder/Strided/MSE_Loss/Model_v2_boxTrue_PretrainedFalse_SigmoidFalse_bs20_lr0.001_dr0.1_every10_dim512/'
            elif opt.dim == 1024 and not(opt.use_7D_feat):
                model_path = 'trained_models/GraphEncoder_Models/Channel25Decoder/Strided/MSE_Loss/Model_v2_boxTrue_PretrainedFalse_SigmoidFalse_bs20_lr0.001_dr0.1_every10_dim1024/'
            elif opt.dim == 1024 and opt.use_7D_feat:
                model_path = 'trained_models/GraphEncoder_Models/Channel25Decoder/Strided/MSE_Loss/Model_v2_boxTrue_PretrainedFalse_SigmoidFalse_bs20_lr0.001_dr0.1_every10_gclip0.1_dimR1024_readoutattend_7DfeatTrue/'
            elif opt.dim ==  2048 and not(opt.use_7D_feat):
                model_path = 'trained_models/GraphEncoder_Models/Channel25Decoder/Strided/MSE_Loss/Model_v2_boxTrue_PretrainedFalse_SigmoidFalse_bs20_lr0.001_dr0.1_every10_dim2048/'
            elif opt.dim == 2048 and opt.use_7D_feat:
                model_path = 'trained_models/GraphEncoder_Models/Channel25Decoder/Strided/MSE_Loss/Model_v2_boxTrue_PretrainedFalse_SigmoidFalse_bs20_lr0.001_dr0.1_every10_gclip0.1_dimR2048_readoutattend_7DfeatTrue/'
                        
        elif opt.decoder_model == 'strided' and opt.use_directed_graph == True:
            if opt.dim == 128:
                model_path = 'trained_models/GraphEncoder_Models_DirectedGraph/Channel25Decoder/Strided/MSE_Loss/Model_v2_boxTrue_PretrainedFalse_SigmoidFalse_bs12_lr0.001_dr0.1_every10_dim128/'
            elif opt.dim == 256:
                model_path = 'trained_models/GraphEncoder_Models_DirectedGraph/Channel25Decoder/Strided/MSE_Loss/Model_v2_boxTrue_PretrainedFalse_SigmoidFalse_bs12_lr0.001_dr0.1_every10_dim256/'
            elif opt.dim == 512:
                model_path = 'trained_models/GraphEncoder_Models_DirectedGraph/Channel25Decoder/Strided/MSE_Loss/Model_v2_boxTrue_PretrainedFalse_SigmoidFalse_bs12_lr0.001_dr0.1_every10_dim512/'
            elif opt.dim == 1024 and  not(opt.use_7D_feat):
                model_path = 'trained_models/GraphEncoder_Models_DirectedGraph/Channel25Decoder/Strided/MSE_Loss/Model_v2_boxTrue_PretrainedFalse_SigmoidFalse_bs12_lr0.001_dr0.1_every10_dim1024/'  
            elif opt.dim == 1024 and opt.use_7D_feat:
                model_path = '/home/dipu/codes/GraphEncoding-RICO/trained_models/GraphEncoder_Models_DirectedGraph/Channel25Decoder/Strided/MSE_Loss/Model_v2_boxTrue_PretrainedFalse_SigmoidFalse_bs12_lr0.001_dr0.1_every10_gclip0.1_dimR1024_readoutattend_7DfeatTrue/'
            elif opt.dim ==  2048 and not(opt.use_7D_feat):
                model_path = 'trained_models/GraphEncoder_Models_DirectedGraph/Channel25Decoder/Strided/MSE_Loss/Model_v2_boxTrue_PretrainedFalse_SigmoidFalse_bs12_lr0.001_dr0.1_every10_dim2048/'
            elif opt.dim == 2048 and opt.use_7D_feat:
                model_path = 'trained_models/GraphEncoder_Models_DirectedGraph/Channel25Decoder/Strided/MSE_Loss/Model_v2_boxTrue_PretrainedFalse_SigmoidFalse_bs12_lr0.001_dr0.1_every10_gclip0.1_dimR2048_readoutattend_7DfeatTrue/'
        #import copy
        print('\n Loading from the Pretrained Network from... ')
        print(model_path)
        model_file = model_path + 'ckp_ep20.pth.tar' 
       
        resume = load_checkpoint(model_file)
        model.load_state_dict(resume['state_dict'])  
        model = model.cuda()
        
    
    if opt.loss =='mse':
        criterion = nn.MSELoss()
    elif opt.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([30.0]).cuda())
    elif opt.loss == 'huber':
        criterion = nn.SmoothL1Loss()
    elif opt.loss == 'l1':
        criterion = nn.L1Loss()   
    
    if opt.loss == 'wmse':
        aa = np.load('data/count_n_area_train.npy')
        weights = aa[:,0] * aa[:,1]
        weights = weights /100000
        weights = 1/weights
        weights = torch.Tensor(weights)
        weights = weights.cuda()
        
    optimizer = torch.optim.Adam(model.parameters(), opt.learning_rate)
    
    margin = opt.margin
    lambda_mul = opt.lambda_mul
    dml_loss = TripletLoss(margin=margin)
    dml_loss = dml_loss.cuda()
    
#    if opt.pretrained:
#        optimizer = torch.optim.Adam([
#                {'params': [param for name, param in model.named_parameters() if 'decoder_raster' not in name]}, 
#                {'params': model.decoder_raster.parameters(), 'lr': opt.learning_rate*0.1}], lr=opt.learning_rate)
    
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
   
    boundingBoxes = getBoundingBoxes()
    epoch_done = True
    
    iteration = 0
    epoch = 0
    
    model.train()
    torch.set_grad_enabled(True)
    time_s = time.time()
    while True:
        if epoch_done:
            if epoch in [5,12,17]: 
                decay_factor = opt.learning_rate_decay_rate
                set_lr2(optimizer, decay_factor)
                print('\n', optimizer, '\n') 
                
#            if epoch > opt.learning_rate_decay_start+1 and epoch%opt.learning_rate_decay_every==0 and opt.learning_rate_decay_start >= 0  :
#                decay_factor = opt.learning_rate_decay_rate
#                set_lr2(optimizer, decay_factor)
            
            #if epoch > opt.learning_rate_decay_start+1 and opt.learning_rate_decay_start >= 0  :
            #     frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
            #     decay_factor = opt.learning_rate_decay_rate ** frac
            #     opt.current_lr = opt.learning_rate * decay_factor
            #else:
            #    opt.current_lr = opt.learning_rate
            #set_lr(optimizer, opt.current_lr)       
                
            losses = AverageMeter()
            losses_recon = AverageMeter()
            losses_dml = AverageMeter()

            epoch_done = False
        
        # Load a batch of data from train split
        data =  loader.get_batch('train')
        images_a = data['images_a'].cuda()
        images_p = data['images_p'].cuda()
        images_n = data['images_n'].cuda()
        
        sg_data_a = {key: torch.from_numpy(data['sg_data_a'][key]).cuda() for key in data['sg_data_a']}
        sg_data_p = {key: torch.from_numpy(data['sg_data_p'][key]).cuda() for key in data['sg_data_p']}
        sg_data_n = {key: torch.from_numpy(data['sg_data_n'][key]).cuda() for key in data['sg_data_n']}
        
        #2. Forward model and compute loss
#        torch.cuda.synchronize()   # Waits for all kernels in all streams on a CUDA device to complete. 
        optimizer.zero_grad()
        emb_a, out_a = model(sg_data_a)
        emb_p, out_p = model(sg_data_p)
        emb_n, out_n = model(sg_data_n)
        
        emb_a = F.normalize(emb_a)
        emb_p = F.normalize(emb_p)
        emb_n = F.normalize(emb_n)
        
        if opt.decoder_model == 'strided' or opt.decoder_model == 'strided_dim2688':
            images_a = F.interpolate(images_a, size= [239,111])
            images_p = F.interpolate(images_p, size= [239,111])
            images_n = F.interpolate(images_n, size= [239,111])
            
        if opt.loss == 'wmse':
            loss_a = weighted_mse_loss_2(out_a,images_a,weights)
            loss_p = weighted_mse_loss_2(out_p,images_p,weights)
            loss_n = weighted_mse_loss_2(out_n,images_n,weights)
        else:
            loss_a = criterion(out_a, images_a)
            loss_p = criterion(out_p, images_p)
            loss_n = criterion(out_n, images_n)
            
        recon_loss = torch.mean(torch.stack([loss_a , loss_p , loss_n]))
        ml_loss = dml_loss(emb_a, emb_p, emb_n)
        
        loss = ml_loss + lambda_mul*recon_loss
        
        # 3. Update model
        loss.backward()
        clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()
        torch.cuda.empty_cache()
        losses.update(loss.detach().item())
        losses_recon.update(recon_loss.detach().item())
        losses_dml.update(ml_loss.detach().item())
        
#        torch.cuda.synchronize()

        # Update the iteration and epoch
        iteration += 1
        
        if epoch ==0 and iteration ==1:
            print("Training Started ")
        
        if iteration%1000 == 0:
            elsp_time = (time.time() - time_s)
            print( 'Epoch [%02d] [%05d / %05d] Average_Loss: %.5f    Recon Loss: %.4f  DML Loss: %.4f'%(epoch+1, iteration*opt.batch_size, len(loader), losses.avg, losses_recon.avg, losses_dml.avg ))
            with open(save_dir+'/log.txt', 'a') as f:
                f.write('Epoch [%02d] [%05d / %05d  ] Average_Loss: %.5f  Recon Loss: %.4f  DML Loss: %.4f\n'%(epoch+1, iteration*opt.batch_size, len(loader), losses.avg, losses_recon.avg, losses_dml.avg ))
                f.write('Completed {} images in {}'.format(iteration*opt.batch_size, elsp_time))
                
            
            print('Completed {} images in {}'.format(iteration*opt.batch_size, elsp_time))
            time_s = time.time()
        
            #print( 'Epoch [%02d] [%05d ] Average_Loss: %.3f' % (epoch+1, iteration*opt.batch_size, len(loader)))
        
        if data['bounds']['wrapped']:
            epoch += 1
            epoch_done = True 
            iteration = 0 
            
        #del data, images, sg_data, out, loss 
    
        if (epoch+1) % 5 == 0  and epoch_done:
            state_dict = model.state_dict()  
            
            save_checkpoint({
            'state_dict': state_dict,
            'epoch': (epoch+1)}, is_best=False, fpath=osp.join(save_dir, 'ckp_ep' + str(epoch + 1) + '.pth.tar'))

            perform_tests_dml(model, loader_test, boundingBoxes, model_name, save_dir, epoch)
            
            # set model to training mode again.
            model.train()
            torch.set_grad_enabled(True)
        if epoch > 25:
            break

def perform_tests_dml(model, loader_test, boundingBoxes, model_name, save_dir, ep):
    model.eval()  
    save_file = save_dir + '/result.txt' 
    q_feat, q_fnames = extract_features(model, loader_test, split='query')
    g_feat, g_fnames = extract_features(model, loader_test, split='gallery')
    
    onlyGallery = True
    if not(onlyGallery):
        t_feat, t_fnames = extract_features(model, loader_test, split='train')
        g_feat = np.vstack((g_feat,t_feat))
        g_fnames = g_fnames + t_fnames 

    q_feat = np.concatenate(q_feat)
    g_feat = np.concatenate(g_feat)
    
    distances = cdist(q_feat, g_feat, metric= 'euclidean')
    sort_inds = np.argsort(distances)

    
    #overallMeanIou, overallMeanWeightedIou, classIoU   = get_overall_IOU(boundingBoxes,sort_inds,g_fnames,q_fnames)         
    overallMeanClassIou, overallMeanWeightedClassIou, classwiseClassIoU = get_overall_Classwise_IOU(boundingBoxes,sort_inds,g_fnames,q_fnames, topk = [1,5,10])
    overallMeanAvgPixAcc, overallMeanWeightedPixAcc, classPixAcc = get_overall_pix_acc(boundingBoxes,sort_inds,g_fnames,q_fnames, topk = [1,5,10])     
    
    #iou_aDCG, iou_wDCG, iou_avg_aNdcg, iou_avg_wNdcg = get_overall_ClasswiseIou_ndcg(boundingBoxes,sort_inds,g_fnames,q_fnames) 
    #acc_aDCG, acc_wDCG, acc_avg_aNdcg, acc_avg_wNdcg = get_overall_PixAcc_ndcg(boundingBoxes,sort_inds,g_fnames,q_fnames) 
            
    print('\n\nep:%s'%(ep))
    print(model_name)
    print('GAlleryOnly Flag:', onlyGallery)
    
    print('overallMeanClassIou =  ' + str([ '{:.3f}'.format(x) for x in overallMeanClassIou]))        
    print('overallMeanWeightedClassIou =  ' + str([ '{:.3f}'.format(x) for x in overallMeanWeightedClassIou]))
    print('overallMeanAvgPixAcc =  ' + str([ '{:.3f}'.format(x) for x in overallMeanAvgPixAcc]))
    print('overallMeanWeightedPixAcc =  ' + str([ '{:.3f}'.format(x) for x in overallMeanWeightedPixAcc]))
    
           
    with open(save_file, 'a') as f:
        f.write('\n\ep: {}\n'.format(ep))
        f.write('Model name: {}\n'.format(model_name))
        f.write('GAlleryOnly Flag: {}\n'.format(onlyGallery))     
       
        f.write('overallMeanClassIou =  ' + str([ '{:.3f}'.format(x) for x in overallMeanClassIou]) + '\n')
        f.write('overallMeanWeightedClassIou =  ' + str([ '{:.3f}'.format(x) for x in overallMeanWeightedClassIou]) + '\n')
        f.write('overallMeanAvgPixAcc =  ' + str([ '{:.3f}'.format(x) for x in overallMeanAvgPixAcc]) + '\n')
        f.write('overallMeanWeightedPixAcc =  ' + str([ '{:.3f}'.format(x) for x in overallMeanWeightedPixAcc]) + '\n')
        


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

#def get_model_name(opt):
#    if 
#    model = 

opt = opts_dml.parse_opt()
for arg in vars(opt):
    print(arg, getattr(opt,arg))
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
if __name__ == '__main__':
    main(opt)
    
#dataset = RICO_ComponentDataset(train_uis, data_dir, transform=data_transform)
#train_loader = torch.utils.data.DataLoader(dataset, batch_size= BATCH_SIZE, num_workers=1) #, collate_fn=lambda x: x[0]

#for i , (data) in enumerate(train_loader):
#    print('i = \n', i)
#    print('wait')

    
#data_iter = iter(train_loader)
#tmp = next(data_iter)
#print('\n Ready .... ')
#print(tmp)

def binary_cross_entropy_with_logits(input, target, weight=None, size_average=None,
                                     reduce=None, reduction='elementwise_mean', pos_weight=None):
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    max_val = (-input).clamp(min=0)

    if pos_weight is None:
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
    else:
        log_weight = 1 + (pos_weight - 1) * target
        loss = input - input * target + log_weight * (max_val + ((-max_val).exp() + (-input - max_val).exp()).log())

    if weight is not None:
        loss = loss * weight

    if reduction == 'none':
        return loss
    elif reduction == 'elementwise_mean':
        return loss.mean()
    else:
        return loss.sum() 
    
    
    