import torch
import torch.nn as nn

from functools import reduce
from operator import mul

class ListModule(nn.Module):
    def __init__(self,*args):
        super(ListModule,self).__init__()
        idx=0
        for module in args:
            self.add_module(str(idx),module)
            idx+=1
            
    def __getitem__(self,idx:int):
        if idx<0 or idx>=len(self._modules):
            raise IndexError("index {} is out of range!!!".format(idx))
        it=iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)
        
    def __iter__(self):
        return iter(self._modules.values())
        
    def __len__(self):
        return len(self._modules)

class MaskedConv2d(nn.Conv2d):
    """
    Implements a Masked Convolution 2D.
    This is a 2D convolution with a masked kernel.
    """
    def __init__(self,mask_type:str,idx:int,*args,**kwargs):
        super(MaskedConv2d,self).__init__(*args,**kwargs)
        assert mask_type in ["A","B"]
        self.register_buffer("mask",self.weight.data.clone())
        _,_,kt,kd=self.weight.size()
        assert kt==3
        self.mask.fill_(0)
        self.mask[:,:,:kt//2,:]=1
        if idx+(mask_type=="B")>0:
            self.mask[:,:,kt//2,:idx+(mask_type=="B")]=1
        self.weight.mask=self.mask
        
    def forward(self,x):
        """
        :param x: the input tensor.
        :return: the output tensor as result of the convolution.
        """
        self.weight.data*=self.mask
        return super(MaskedConv2d,self).forward(x)

class MaskedStackedConvolution(torch.nn.Module):
    """
    Implements a Masked Stacked Convolution layer.
    The autoregressive layer emplyed for the estimation of densities of video feature vectors.
    """
    def __init__(self,mask_type,feature_dim,in_channels,out_channels):
        """
        :param mask_type: type of autoregressive layer, either "A" or "B".
        :param feature_dim: the length of each feature vector in the time series.
        :param in_channels: number of input channels.
        :param out_channels: number of output channels.
        """
        super(MaskedStackedConvolution,self).__init__()
        self.mask_type=mask_type
        self.feature_dim=feature_dim
        self.in_channels=in_channels
        self.out_channels=out_channels
        
        layers=[]
        for i in range(0,feature_dim):
            layers.append(MaskedConv2d(mask_type=mask_type,
                                        idx=i,
                                        in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=(3,feature_dim),
                                        padding=(1,0)))
        self.conv_layers=ListModule(*layers)
    
    def forward(self,x):
        """
        :param x: the input tensor.
        :return: the output of a MSC manipulation.
        """
        #print("##################model/cpd_layer.py, MaskedStackedConvolution, x.shape:{}".format(x.size()))
        out=[]
        for i in range(0,self.feature_dim):
            out.append(self.conv_layers[i](x))
            #print("##################model/cpd_layer.py, MaskedStackedConvolution, out[0].shape:{}".format(out[0].size()))
        out=torch.cat(out,dim=-1)
        #print("#####################model/cpd_layer.py, MaskedStackedConvolution, out.shape:{}".format(out.size()))
        return out
        
    def __repr__(self):
        return self.__class__.__name__+"("\
                +"mask_type="+str(self.mask_type)\
                +", feature_dim="+str(self.feature_dim)\
                +", in_channels="+str(self.in_channels)\
                +", out_channels="+str(self.out_channels)\
                +", n_params="+str(self.n_parameters)+")"
                
    @property
    def n_parameters(self):
        n_parameters=0
        for p in self.parameters():
            if hasattr(p,"mask"):
                n_parameters+=torch.sum(p.mask).item()
            else:
                n_parameters+=reduce(mul,p.shape)
        return int(n_parameters)

class Estimator2D(torch.nn.Module):
    """
    Implements an estimator for 2-dimensional vector.
    2-dimensional vectors arise from the encoding of video clips.
    Takes as input a time series of latent vectors and outputs cpds for each variable.
    """
    def __init__(self,feature_dim,fm_list,cpd_channels):
        """
        :param feature_dim: the dimensionality of latent vectors.
        :param fm_list: list of channels for each MFC layer.
        :param cpd_channels: numbers of bins in which the multinomial works.
        """
        super(Estimator2D,self).__init__()
        self.feature_dim=feature_dim
        self.fm_list=fm_list
        self.cpd_channels=cpd_channels
        activation_fn=nn.LeakyReLU()
        
        layers_list=[]
        mask_type="A"
        fm_in=1
        for l in range(0,len(fm_list)):
            fm_out=fm_list[l]
            layers_list.append(MaskedStackedConvolution(mask_type=mask_type,feature_dim=feature_dim,in_channels=fm_in,out_channels=fm_out))
            layers_list.append(activation_fn)
            mask_type="B"
            fm_in=fm_list[l]
        layers_list.append(MaskedStackedConvolution(mask_type=mask_type,feature_dim=feature_dim,in_channels=fm_in,out_channels=cpd_channels))
        self.layers=nn.Sequential(*layers_list)
        
    def forward(self,x):
        """
        :param x: the batch of latent vectors.
        :return: the batch of CPD estimates.
        """
        #print("####################model/cpd_layer.py, Estimator, x.size():{}".format(x.size()))
        h=torch.unsqueeze(x,dim=1)
        h=self.layers(h)
        o=h
        #print("####################model/cpd_layer.py, Estimator, o.size():{}".format(o.size()))
        return o
