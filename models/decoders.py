#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 18:45:02 2020

@author: dipu
"""


import torch.nn as nn

def raster_decoder():
    decoder = nn.Sequential(
       nn.Upsample(scale_factor =2, mode = 'nearest'),
       nn.ConvTranspose2d(32,16,3),
       nn.ReLU(),
       
       nn.Upsample(scale_factor =2, mode = 'nearest'),
       nn.ConvTranspose2d(16,16,3),
       nn.ReLU(),
       
       nn.Upsample(scale_factor =2, mode = 'nearest'),
       nn.ConvTranspose2d(16,8,3),
       nn.ReLU(),
       
       nn.Upsample(scale_factor =2, mode = 'nearest'),
       nn.ConvTranspose2d(8,3,3),
       nn.ReLU(),
       )
    return decoder


def decoder_25Channel():
    decoder = nn.Sequential(
            nn.Upsample(scale_factor =2, mode = 'nearest'),
            nn.ConvTranspose2d(32,25,3),
            nn.ReLU(),
           
            nn.Upsample(scale_factor =2, mode = 'nearest'),
            nn.ConvTranspose2d(25,25,3),
            nn.ReLU(),
           
            nn.Upsample(scale_factor =2, mode = 'nearest'),
            nn.ConvTranspose2d(25,25,3),
            nn.ReLU(),
           
            nn.Upsample(scale_factor =2, mode = 'nearest'),
            nn.ConvTranspose2d(25,25,3),
            nn.ReLU(),
            )
     
    return decoder

def decoder_25Channel_convOnly():
    decoder = nn.Sequential(
            nn.ConvTranspose2d(32,25,3,  stride=2),
            nn.ReLU(),
            
            nn.ConvTranspose2d(25,25,3, stride=2),
            nn.ReLU(),
           
            nn.ConvTranspose2d(25,25,3, stride=2),
            nn.ReLU(),
           
            nn.ConvTranspose2d(25,25,3, stride=2),
            nn.ReLU(),
            )
     
    return decoder
