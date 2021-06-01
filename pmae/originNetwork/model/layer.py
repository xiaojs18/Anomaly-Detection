import torch
import torch.nn as nn

def residual_op(x,functions,bns,activation_fn):
    """
    implements a global residual operation
    
    :param x: the input tensor
    :param functions: a list of functions
    :param bns: a list of optional batch-norm layers
    :param activation_fn: the activation to be applied
    :return: the output of the residual operation
    """
    f1,f2,f3=functions
    bn1,bn2,bn3=bns
    assert len(functions)==len(bns)==3
    assert f1 is not None and f2 is not None
    assert not (f3 is None and bn3 is not None)
    
    #A-branch
    ha=x
    ha=f1(ha)
    if bn1 is not None:
        ha=bn1(ha)
    ha=activation_fn(ha)
    
    ha=f2(ha)
    if bn2 is not None:
        ha=bn2(ha)
        
    #B-branch
    hb=x
    if f3 is not None:
        hb=f3(hb)
    if bn3 is not None:
        hb=bn3(hb)
        
    out=ha+hb
    return activation_fn(out)

class MaskedConv3d(nn.Conv3d):
    """
    3D convolution that cannot access future frames
    """
    def __init__(self,*args,**kwargs):
        super(MaskedConv3d,self).__init__(*args,**kwargs)
        self.register_buffer("mask",self.weight.data.clone())
        _,_,kt,kh,kw=self.weight.size()
        self.mask.fill_(1)
        self.mask[:,:,kt//2+1:]=0
        
    def forward(self,x):
        self.weight.data*=self.mask
        return super(MaskedConv3d,self).forward(x)

class DownsampleBlock(torch.nn.Module):
    def __init__(self,channel_in,channel_out,activation_fn,stride,use_bn=True,use_bias=False):
        """
        downsample block.
        
        :param channel_in: number of input channels
        :param channel_out: number of output channels
        :param activation_fn: activation to be employed
        :param stride: the stride to be applied to downsample feature maps
        :param use_bn: whether or not to use batch-norm
        :param use_bias: whether or nor to use bias
        """
        super(DownsampleBlock,self).__init__()
        self.stride=stride
        self._channel_in=channel_in
        self._channel_out=channel_out
        self._activation_fn=activation_fn
        self._use_bn=use_bn
        self._bias=use_bias
        self.conv1a=MaskedConv3d(in_channels=channel_in,out_channels=channel_out,kernel_size=3,
                                padding=1,stride=stride,bias=use_bias)
        self.conv1b=MaskedConv3d(in_channels=channel_out,out_channels=channel_out,kernel_size=3,
                                padding=1,stride=1,bias=use_bias)
        self.conv2a=nn.Conv3d(in_channels=channel_in,out_channels=channel_out,kernel_size=1,
                                padding=0,stride=stride,bias=use_bias)
        self.bn1a=self.get_bn()
        self.bn1b=self.get_bn()
        self.bn2a=self.get_bn()
        
    def get_bn(self):
        return nn.BatchNorm3d(num_features=self._channel_out) if self._use_bn else None
    
    def forward(self,x):
        return residual_op(
            x,
            functions=[self.conv1a,self.conv1b,self.conv2a],
            bns=[self.bn1a,self.bn1b,self.bn2a],
            activation_fn=self._activation_fn
        )
        
class TemporallySharedFullyConnection(torch.nn.Module):
    def __init__(self,in_features,out_features,bias=True):
        """
        implements a temporally-shared fully connection
        process a time series of feature vectors
        performs the same linear projection to all of them
        
        :param in_features: number of input features
        :param out_features: number of output features
        :param bias: whether or not to add bias
        """
        super(TemporallySharedFullyConnection,self).__init__()
        self.in_features=in_features
        self.out_features=out_features
        self.bias=bias
        self.linear=nn.Linear(in_features=in_features,out_features=out_features,bias=bias)
        
    def forward(self,x):
        b,t,d=x.size()
        output=[]
        for i in range(0,t):
            output.append(self.linear(x[:,i,:]))
            #print("##################smy model/layer/tsc output[0].shape:{}".format(output[0].shape))
        output=torch.stack(output,1)
        #print("############smy model/layer/tsc output.shape:{}".format(output.shape))
        return output
        
class UpsampleBlock(torch.nn.Module):
    def __init__(self,channel_in,channel_out,activation_fn,stride,output_padding,use_bn=True,use_bias=False):
        """
        upsample block.
        
        :param channel_in: number of input channels
        :param channel_out: number of output channels
        :param activation_fn: activation to be employed
        :param stride: the stride to be applied to downsample feature maps
        :param output_padding: the padding to be added applied output feature maps
        :param use_bn: whether or not to use batch-norm
        :param use_bias: whether or nor to use bias
        """
        super(UpsampleBlock,self).__init__()
        self.stride=stride
        self._channel_in=channel_in
        self._channel_out=channel_out
        self._activation_fn=activation_fn
        self._use_bn=use_bn
        self._bias=use_bias
        self.output_padding=output_padding
        self.conv1a=nn.ConvTranspose3d(channel_in,channel_out,kernel_size=5,padding=2,stride=stride,output_padding=output_padding,bias=use_bias)
        self.conv1b=nn.Conv3d(in_channels=channel_out,out_channels=channel_out,kernel_size=3,padding=1,stride=1,bias=use_bias)
        self.conv2a=nn.ConvTranspose3d(channel_in,channel_out,kernel_size=5,padding=2,stride=stride,output_padding=output_padding,bias=use_bias)
        self.bn1a=self.get_bn()
        self.bn1b=self.get_bn()
        self.bn2a=self.get_bn()
        
    def get_bn(self):
        return nn.BatchNorm3d(num_features=self._channel_out) if self._use_bn else None
        
    def forward(self,x):
        return residual_op(
            x,
            functions=[self.conv1a,self.conv1b,self.conv2a],
            bns=[self.bn1a,self.bn1b,self.bn2a],
            activation_fn=self._activation_fn
        )
