import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layer import DownsampleBlock,TemporallySharedFullyConnection,UpsampleBlock
from model.cpd_layer import Estimator2D
from model.mem_layer import Memory

class Encoder(torch.nn.Module):
    def __init__(self,t_length=5,n_channel=3,feature_dim=512,h=256,w=256):
        super(Encoder,self).__init__()
        activation_fn=nn.LeakyReLU()
        '''
        self.conv=nn.Sequential(
            DownsampleBlock(channel_in=n_channel,channel_out=8,activation_fn=activation_fn,stride=(1,2,2)),
            DownsampleBlock(channel_in=8,channel_out=16,activation_fn=activation_fn,stride=(1,2,2)),
            DownsampleBlock(channel_in=16,channel_out=32,activation_fn=activation_fn,stride=(2,2,2)),
            DownsampleBlock(channel_in=32,channel_out=64,activation_fn=activation_fn,stride=(1,2,2)),
            DownsampleBlock(channel_in=64,channel_out=64,activation_fn=activation_fn,stride=(2,2,2))
        )
        '''
        self.encoder_conv1=nn.Sequential(DownsampleBlock(channel_in=n_channel,channel_out=8,activation_fn=activation_fn,stride=(1,2,2)))
        self.encoder_conv2=nn.Sequential(DownsampleBlock(channel_in=8,channel_out=16,activation_fn=activation_fn,stride=(1,2,2)))
        self.encoder_conv3=nn.Sequential(DownsampleBlock(channel_in=16,channel_out=32,activation_fn=activation_fn,stride=(2,2,2)))
        self.encoder_conv4=nn.Sequential(DownsampleBlock(channel_in=32,channel_out=64,activation_fn=activation_fn,stride=(1,2,2)))
        self.encoder_conv5=nn.Sequential(DownsampleBlock(channel_in=64,channel_out=64,activation_fn=activation_fn,stride=(2,2,2)))
        self.deepest_shape=(64,(t_length-1)//4,h//32,w//32)
        #self.deepest_shape=(64,t_length-1,h//32,w//32)
        dc,dt,dh,dw=self.deepest_shape
        self.tdl=nn.Sequential(
            TemporallySharedFullyConnection(in_features=(dc*dh*dw),out_features=512),
            nn.Tanh(),
            TemporallySharedFullyConnection(in_features=512,out_features=feature_dim),
            nn.Sigmoid()
        )
        
    def forward(self,x):
        h=x
        #h=self.conv(h)
        #print("############model/encoder_backbone.py encoder deepest_shape.shape:{}".format(self.deepest_shape))
        #print("############model/encoder_backbone.py encoder h0.shape:{}".format(h.shape))
        h1=self.encoder_conv1(h)
        #print("############model/encoder_backbone.py encoder h1.shape:{}".format(h1.shape))
        h2=self.encoder_conv2(h1)
        #print("############model/encoder_backbone.py encoder h2.shape:{}".format(h2.shape))
        h3=self.encoder_conv3(h2)
        #print("############model/encoder_backbone.py encoder h3.shape:{}".format(h3.shape))
        h4=self.encoder_conv4(h3)
        #print("############model/encoder_backbone.py encoder h4.shape:{}".format(h4.shape))
        h5=self.encoder_conv5(h4)
        #print("############model/encoder_backbone.py encoder h5.shape:{}".format(h5.shape))
        c,t,height,width=self.deepest_shape
        h=torch.transpose(h5,1,2).contiguous()
        #print("############model/encoder_backbone.py encoder h.shape:{}".format(h.shape))
        h=h.view(-1,t,(c*height*width))
        o=self.tdl(h)
        return o,h1,h2,h3,h4,h5

class Decoder(torch.nn.Module):
    def __init__(self,deepest_shape,t_length=5,n_channel=3,feature_dim=512,h=256,w=256):
        super(Decoder,self).__init__()
        self.feature_dim=feature_dim
        self.deepest_shape=deepest_shape
        dc,dt,dh,dw=deepest_shape
        activation_fn=nn.LeakyReLU()
        self.tdl=nn.Sequential(
            TemporallySharedFullyConnection(in_features=feature_dim,out_features=512),
            nn.Tanh(),
            TemporallySharedFullyConnection(in_features=512,out_features=(dc*dh*dw)),
            activation_fn
        )
        '''
        self.conv=nn.Sequential(
            UpsampleBlock(channel_in=dc,channel_out=64,activation_fn=activation_fn,stride=(2,2,2),output_padding=(1,1,1)),
            UpsampleBlock(channel_in=64,channel_out=32,activation_fn=activation_fn,stride=(1,2,2),output_padding=(0,1,1)),
            UpsampleBlock(channel_in=32,channel_out=16,activation_fn=activation_fn,stride=(2,2,2),output_padding=(1,1,1)),
            UpsampleBlock(channel_in=16,channel_out=8,activation_fn=activation_fn,stride=(1,2,2),output_padding=(0,1,1)),
            UpsampleBlock(channel_in=8,channel_out=8,activation_fn=activation_fn,stride=(1,2,2),output_padding=(0,1,1)),
            nn.Conv3d(in_channels=8,out_channels=n_channel,kernel_size=1)
        )
        '''
        self.decoder_conv1=nn.Sequential(UpsampleBlock(channel_in=dc*3,channel_out=64,activation_fn=activation_fn,stride=(2,2,2),output_padding=(1,1,1)))
        self.decoder_conv2=nn.Sequential(UpsampleBlock(channel_in=64*2,channel_out=32,activation_fn=activation_fn,stride=(1,2,2),output_padding=(0,1,1)))
        self.decoder_conv3=nn.Sequential(UpsampleBlock(channel_in=32*2,channel_out=16,activation_fn=activation_fn,stride=(2,2,2),output_padding=(1,1,1)))
        self.decoder_conv4=nn.Sequential(UpsampleBlock(channel_in=16*2,channel_out=8,activation_fn=activation_fn,stride=(1,2,2),output_padding=(0,1,1)))
        self.decoder_conv5=nn.Sequential(UpsampleBlock(channel_in=8*2,channel_out=8,activation_fn=activation_fn,stride=(1,2,2),output_padding=(0,1,1)))
        self.decoder_conv6=nn.Sequential(nn.Conv3d(in_channels=8,out_channels=n_channel,kernel_size=(t_length-1,1,1)))
        
        
    def forward(self,x,updated_x,h1,h2,h3,h4,h5):
        dc,dt,dh,dw=self.deepest_shape
        
        h=x
        h=self.tdl(h)
        h=h.view(len(h),dt,dc,dh,dw)
        h=torch.transpose(h,1,2).contiguous()
        
        updated_x=torch.mean(updated_x,dim=1)
        updated_h=updated_x.view_as(x)
        updated_h=self.tdl(updated_h)
        updated_h=updated_h.view(len(updated_h),dt,dc,dh,dw)
        updated_h=torch.transpose(updated_h,1,2).contiguous()
        
        #print("############model/encoder_backbone.py decoder h0.shape:{}".format(h.shape))
        #########h=self.conv(h)
        h = torch.cat((updated_h, h), dim = 1)
        h = torch.cat((h5, h), dim = 1)
        h=self.decoder_conv1(h)
        #print("############model/encoder_backbone.py decoder h1.shape:{}".format(h.shape))
        h = torch.cat((h4, h), dim = 1)
        h=self.decoder_conv2(h)
        #print("############model/encoder_backbone.py decoder h2.shape:{}".format(h.shape))
        h = torch.cat((h3, h), dim = 1)
        h=self.decoder_conv3(h)
        #print("############model/encoder_backbone.py decoder h3.shape:{}".format(h.shape))
        h = torch.cat((h2, h), dim = 1)
        h=self.decoder_conv4(h)
        #print("############model/encoder_backbone.py decoder h4.shape:{}".format(h.shape))
        h = torch.cat((h1, h), dim = 1)
        h=self.decoder_conv5(h)
        #print("############model/encoder_backbone.py decoder h5.shape:{}".format(h.shape))
        h=self.decoder_conv6(h)
        #print("############model/encoder_backbone.py decoder h6.shape:{}".format(h.shape))
        o=h
        return o
        

class EncoderBackbone(torch.nn.Module):
    def __init__(self,cpd_channels=100,n_channel=3,t_length=5,feature_dim=512,memory_size=10,memory_dim=512):
        super(EncoderBackbone,self).__init__()
        self.feature_dim=feature_dim
        self.cpd_channels=cpd_channels
        self.encoder=Encoder(t_length,n_channel,feature_dim=512)
        self.decoder=Decoder(t_length=t_length,deepest_shape=self.encoder.deepest_shape,n_channel=n_channel,feature_dim=512)
        self.estimator=Estimator2D(feature_dim=feature_dim,fm_list=[4,4],cpd_channels=cpd_channels)
        self.memory=Memory(memory_size,feature_dim,memory_dim)
        
    def forward(self,x,mems,train=True):
        h=x
        z,h1,h2,h3,h4,h5=self.encoder(h)
        z_dist=self.estimator(z)
        if train:
            updated_z_dist,mems,score_query,score_memory,separateness_loss,compactness_loss=self.memory(z_dist,mems,train)
            #decoder_z = torch.cat((z, updated_z_dist), dim = 1)
            x_r=self.decoder(z,updated_z_dist,h1,h2,h3,h4,h5)
            return x_r,z,z_dist,mems,compactness_loss,separateness_loss,score_query,score_memory
        else:
            updated_z_dist,mems,score_query,score_memory,query,top1_key,key_indice,compactness_loss=self.memory(z_dist,mems,train)
            x_r=self.decoder(z,updated_z_dist,h1,h2,h3,h4,h5)
            return x_r,z,z_dist,mems,compactness_loss,query,top1_key,key_indice,score_query,score_memory
