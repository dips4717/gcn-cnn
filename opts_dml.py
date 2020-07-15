#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 18:35:09 2020

@author: dipu
"""

import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise Exception("value not allowed")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, default = 'test',
                        help = 'an id identifying this run/job used in cross-val and apppended when writing progress files')
    parser.add_argument('--gpu_id', type=str, default = '3', help = 'GPU ID')
    
    # DML Parameters
    parser.add_argument('--margin', type=float, default=0.2, # .0, #5.,
                    help='margin parameter of triplet loss')
    parser.add_argument('--lambda_mul', type=float, default=10.0, # .0, #5.,
                    help='margin parameter of triplet loss')
    
    # Directories
    parser.add_argument('--img_dir', type=str, default='/mnt/amber/scratch/Dipu/RICO/semantic_annotations/',  
                    help='path to the semantic UI images of RICO dataset')
    parser.add_argument('--Channel25_img_dir', type=str, default='/mnt/amber/scratch/Dipu/RICO/25ChannelImages',  
                    help='path to the precomputed 25 Channel image representation of RICO UIs')
    
    
    # Model parameters 
    parser.add_argument('--decoder_model', type=str, default = 'strided',  # 'strided',#'upsample',  #
                        help='which decoder upsample or strided')
    parser.add_argument('--last_layer_sigmoid', type=str2bool, default = False, 
                        help='whether to sigmoid activation after last layer')
    parser.add_argument('--pretrained', type=str2bool, default= True, 
                        help = 'where to initalize the weights from pretrained models/ warm-starting')
    parser.add_argument('--dim', type= int, default = 1024,
                        help = 'dimension of the latent embedding 512 or 2688')
    parser.add_argument('--readout', type = str, default =  'attend', #'inverse', # 'average', 'inverse', 'attend' 
                        help = 'what operation to be used to readout the nodes')
    
    # Dataloader parameters
    parser.add_argument('--use_directed_graph', type=str2bool, default = True,
                        help='undirected or directed graph')
    parser.add_argument('--use_precomputed_25Chan_imgs', type=str2bool, default =True,  
                        help ='whether to pre-computed 25 Channel Images for faster dataloading/training') 
    parser.add_argument('--use_25_images', type=str2bool, default =True,  
                        help ='whether to use 3-channel Semanti UI or 25 Channel images for loss')
    parser.add_argument('--apn_dict_path', type=str, default ='Triplets/apn_dict_48K_pthres60.pkl', 
                        help='path to the training triplets computed based on IoU')
    parser.add_argument('--xy_modified_feat', type=str2bool, default = False,
                        help='xy shifts normalized by width  & height or by the area')
    parser.add_argument('--containment_feat', type=str2bool, default = False,
                        help='undirected or directed graph')
    parser.add_argument('--use_box_feats', type=str2bool, default = True, 
                        help='whether to use geometric box features')
    parser.add_argument('--use_7D_feat', type=str2bool, default = False, 
                        help='whether to use geometric box features')
    parser.add_argument('--hardmining', type=str2bool, default = False, 
                        help='whether to use geometric box features')
    
    # Optimization: General
    parser.add_argument('--loss', type=str, default='mse',  # 'bce', 'l1', 'huber', # 'wmse'
                        help='loss function to use, mse or bce')
    parser.add_argument('--max_epochs', type=int, default=20,
                    help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=4,  # 10,
                    help='minibatch size')
    parser.add_argument('--grad_clip', type=float, default=1, # .0, #5.,
                    help='clip gradients at this value')
    
     # learning rate
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                    help='learning rate')
    parser.add_argument('--learning_rate_decay_start', type=int, default=0,
                    help='at what epoch to start decaying learning rate? (-1 = dont)')
    parser.add_argument('--learning_rate_decay_every', type=int, default=7, #3
                    help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.1, # 0.5
                    help='every how many iterations thereafter to drop LR?(in epoch)')
    
    parser.add_argument('--rnn_size', type=int, default=1000,
                    help='size of the rnn in number of hidden nodes in each layer')
    
    parser.add_argument('--drop_prob', type=float, default=0.5,
                    help='strength of dropout various layers')
    
    args = parser.parse_args()

    return args