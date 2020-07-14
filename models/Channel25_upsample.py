#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 12:51:35 2019

@author: dipu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
#from decoders import raster_decoder, decoder_25Channel, decoder_25Channel_convOnly


def build_embeding_layer(vocab_size, dim, drop_prob):
    embed = nn.Sequential(nn.Embedding(vocab_size, dim),
                          nn.ReLU(),
                          nn.Dropout(drop_prob))
    return embed

#%%
class GNN(nn.Module):
    def __init__(self, opt):
        super(GNN, self).__init__()
        self.opt = opt
        self.dim = self.opt.dim
        in_dim = opt.rnn_size
        out_dim = opt.rnn_size
        
#        if self.opt.rela_gnn_type==0:
#            in_rela_dim = in_dim*3
#        elif self.opt.rela_gnn_type==1:
#            in_rela_dim = in_dim*2
#        else:
#            raise NotImplementedError()

        in_rela_dim = in_dim*3
    
        # gnn with simple MLP
#        self.gnn_attr = nn.Sequential(nn.Linear(in_dim*2, out_dim),
#                                        nn.ReLU(inplace=True),
#                                        nn.Dropout(opt.drop_prob_lm))
        
        self.gnn_rela = nn.Sequential(nn.Linear(in_rela_dim, out_dim),
                                          nn.ReLU(inplace=True),
                                          nn.Dropout(opt.drop_prob))
        
        
        
    def forward(self, obj_vecs, rela_vecs, edges, rela_masks=None):
        # for easily indexing the subject and object of each relation in the tensors
#        obj_vecs, attr_vecs, rela_vecs, edges, ori_shape = self.feat_3d_to_2d(obj_vecs, attr_vecs, rela_vecs, edges)
        obj_vecs, rela_vecs, edges, ori_shape = self.feat_3d_to_2d(obj_vecs,  rela_vecs, edges)
        rela_masks = rela_masks.unsqueeze(-1)
        # obj
        new_obj_vecs = obj_vecs

        # attr
#        new_attr_vecs = self.gnn_attr(torch.cat([obj_vecs, attr_vecs], dim=-1)) + attr_vecs

        # rela
        # get node features for each triplet <subject, relation, object>
        s_idx = edges[:, 0].contiguous() # index of subject
        o_idx = edges[:, 1].contiguous() # index of object
        s_vecs = obj_vecs[s_idx]
        o_vecs = obj_vecs[o_idx]
        
        t_vecs = torch.cat([s_vecs, rela_vecs, o_vecs], dim=1)

#        new_rela_vecs = self.gnn_rela(t_vecs)+rela_vecs
        new_rela_vecs = self.gnn_rela(t_vecs)
        
        new_obj_vecs,  new_rela_vecs = self.feat_2d_to_3d(new_obj_vecs,  new_rela_vecs, rela_masks, ori_shape)
        return new_obj_vecs,  new_rela_vecs


    def feat_3d_to_2d(self, obj_vecs, rela_vecs, edges):
        """
        convert 3d features of shape (B, N, d) into 2d features of shape (B*N, d)
        """
        B, No = obj_vecs.shape[:2]
        obj_vecs = obj_vecs.view(-1, obj_vecs.size(-1))
#        attr_vecs = attr_vecs.view(-1, attr_vecs.size(-1))
        rela_vecs = rela_vecs.view(-1, rela_vecs.size(-1))

        # edge: (B, max_rela_num, 2) => (B*max_rela_num, 2)
        obj_offsets = edges.new_tensor(range(0, B * No, No))
        edges = edges + obj_offsets.view(-1, 1, 1)
        edges = edges.view(-1, edges.size(-1))
        return obj_vecs, rela_vecs, edges, (B, No)


    def feat_2d_to_3d(self, obj_vecs, rela_vecs, rela_masks, ori_shape):
        """
        convert 2d features of shape (B*N, d) back into 3d features of shape (B, N, d)
        """
        B, No = ori_shape
        obj_vecs = obj_vecs.view(B, No, -1)
#        attr_vecs = attr_vecs.view(B, No, -1)
        rela_vecs = rela_vecs.view(B, -1, rela_vecs.size(-1)) * rela_masks
        return obj_vecs, rela_vecs
    
#%%    
class GraphEncoder_25ChanUpSampleDecoder(nn.Module):
    def __init__(self,opt):
        super(GraphEncoder_25ChanUpSampleDecoder, self).__init__()
        self.opt = opt
        self.dim = self.opt.dim
        self.geometry_relation = True
        self.drop_prob = 0.5
        if self.opt.containment_feat:
            self.geom_feat_size = 9 
        else:
            self.geom_feat_size = 8
        
        
        num_objs = 26 #Our label starts from 1 to 25, nn.embeddings starts from 0 index
        self.obj_embed = build_embeding_layer(num_objs, 128, self.drop_prob)
        
        if self.opt.use_box_feats:
            self.proj_obj_feats = nn.Sequential(*[nn.Linear(5,128), nn.ReLU(), nn.Dropout(0.5)])
            self.proj_cat_feats = nn.Sequential(*[nn.Linear(128+128,128), nn.ReLU(), nn.Dropout(0.5)]) 
    
        self.proj_obj = nn.Sequential(*[nn.Linear(128,1000), nn.ReLU(), nn.Dropout(0.5)])
        self.proj_rela = nn.Sequential(*[nn.Linear(self.geom_feat_size,1000), nn.ReLU(), nn.Dropout(0.5)])
        self.gnn = GNN(opt)
        
        self.attention_obj = Attention(opt)
        #self.attention_attr = Attention(opt)
        self.attention_rela = Attention(opt)
        
        self.read_out = nn.Sequential(nn.Linear(2000,self.dim), nn.ReLU())
        
        self.decoder_FC = nn.Linear(self.dim, 32*14*6)
        
        self.decoder = nn.Sequential(
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
        
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.1
#        self.embed[0].weight.data.uniform_(-initrange, initrange)
        self.obj_embed[0].weight.data.uniform_(-initrange, initrange)
        # self.attr_embed[0].weight.data.uniform_(-initrange, initrange)
        
    def _decoder(self, x):
        x = self.decoder_FC(x)
        x = x.reshape(x.size(0),32,14,6)
        x = self.decoder_raster(x)
        return x    
        
    def forward(self, sg_data):
        #Encoder section
        obj_labels = sg_data['obj_labels']
        obj_masks  = sg_data['obj_masks']
        rela_masks = sg_data['rela_masks']
        rela_edges = sg_data['rela_edges']
        rela_feats = sg_data['rela_feats']
        
#        obj_masks = obj_masks.unsqueeze(-1)
#        rela_masks = rela_masks.unsqueeze(-1)
        
        #get the embeddings 
        obj_embed = self.obj_embed(obj_labels)
        rela_embed = rela_feats
        
        #project the embeddings
        if self.opt.use_box_feats:
            obj_feats =  sg_data['box_feats']
            obj_feats = self.proj_obj_feats(obj_feats)
            obj_embed = obj_embed.squeeze(2)
            obj_cat_feat = torch.cat((obj_feats, obj_embed), 2)
            obj_embed = self.proj_cat_feats(obj_cat_feat)
            
        obj_vecs = self.proj_obj(obj_embed) 
        rela_vecs = self.proj_rela(rela_embed)
        
        #Apply GCN to 
        obj_vecs, rela_vecs = self.gnn(obj_vecs, rela_vecs, rela_edges, rela_masks)
        
        #Get the summarized vectors using attention
        obj_att_vec = self.attention_obj(obj_vecs, att_masks = obj_masks)
        rela_att_vec = self.attention_rela(rela_vecs, att_masks = rela_masks)
        
#        ============================================================
#        #Get the summarized vectors using attention/average/inverse.
#        if self.opt.readout == 'attend':
#            obj_att_vec = self.attention_obj(obj_vecs, att_masks = obj_masks)
#            rela_att_vec = self.attention_rela(rela_vecs, att_masks = rela_masks)
#       
#        elif self.opt.readout == 'average':
#            obj_att_vec = torch.sum(obj_vecs, dim =1) / torch.sum(obj_masks, dim =1, keepdim= True)
#            rela_att_vec = torch.sum(rela_vecs, dim =1) / torch.sum(rela_masks, dim =1, keepdim = True)
#        
#        elif self.opt.readout == 'inverse':
#            boxes = sg_data['obj_boxes']
#            areas = boxes[:,:,2] * boxes[:,:,3]
#            weight = areas/ torch.sum(areas,dim=1).view(areas.shape[0],1)
#            weight = 1/weight 
#            weight[weight==float('inf')] = 0
#            weight = weight / weight.sum(1, keepdim=True)
#            obj_att_vec = torch.bmm(weight.unsqueeze(1), obj_vecs).squeeze(1)
#        
#            # Relation is still average:  Todo: inverse of average of the areas of the boxes that edges connects.
#            rela_att_vec = torch.sum(rela_vecs, dim =1) / torch.sum(rela_masks, dim =1, keepdim = True)
#            
#        else:
#            raise Exception('readout method no implemented')
#       =================================================================  
          
#            
#            
#                        # if self.opt.readout == 'average':
#    else:
#        if self.opt.readout == 'inverse':
#            
            
        # Get the summarized vectors using average
        # obj_att_vec = torch.sum(obj_vecs, dim =1) / torch.sum(obj_masks, dim =1).view(obj_masks.shape[0],1)
        # Get the summarized vectors using weighted average, where weight is inverse of the areas of boxes
        

        # Encoded vector



        enc_vec = self.read_out(torch.cat([obj_att_vec, rela_att_vec],1))
        
        #Decoder section: Decoder 
        x_rec = self._decoder(enc_vec)
        if self.opt.last_layer_sigmoid:
            x_rec = F.sigmoid(x_rec)
        
        return enc_vec, x_rec
        
#%%    
class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.opt = opt
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.rnn_size
#        self.query_dim = self.rnn_size
#        self.h2att = nn.Linear(self.query_dim, self.att_hid_size)
        self.alpha_net = nn.Linear(self.rnn_size, 1)

#    def forward(self, h, att_feats, p_att_feats, att_masks=None):
    def forward(self, att_feats,  att_masks=None):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
#        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        
#        att_h = self.h2att(h)                        # batch * att_hid_size
#        att_h = att_h.unsqueeze(1).expand_as(att)            # batch * att_size * att_hid_size
#        dot = att + att_h                                   # batch * att_size * att_hid_size
        
        dot = att_feats
        dot = torch.tanh(dot)                                # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)               # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)                           # (batch * att_size) * 1
        dot = dot.view(-1, att_size)                        # batch * att_size

        weight = F.softmax(dot, dim=1)                       # batch * att_size
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).float()
            weight = weight / weight.sum(1, keepdim=True) # normalize to 1
        att_feats_ = att_feats.view(-1, att_size, att_feats.size(-1)) # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # batch * att_feat_size
        return att_res
    
#%%    
