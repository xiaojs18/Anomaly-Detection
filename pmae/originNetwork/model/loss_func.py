import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class ReconstructionLoss(torch.nn.Module):
    """
    Implements the reconstruction loss.
    """
    def __init__(self):
        super(ReconstructionLoss,self).__init__()
        self.r_loss=nn.MSELoss(reduction="none")
        
    def forward(self,x,x_r):
        """
        :param x: the batch of input samples.
        :param x_r: the batch of reconstructions.
        :return: the mean reconstruction loss(averaged along the batch axis).
        """
        """
        L=torch.pow((x-x_r),2)
        while L.dim()>1:
            L=torch.sum(L,dim=-1)
        return torch.mean(L)
        """
        L=self.r_loss(x,x_r)
        return torch.mean(L)

class AutoregressionLoss(torch.nn.Module):
    """
    Implements the autoregression loss.
    Given a representation and the estimated cpds,
    provides the log-likelihood of the representation under the estimated prior.
    """
    def __init__(self,cpd_channels):
        super(AutoregressionLoss,self).__init__()
        self.cpd_channels=cpd_channels
        self.eps=np.finfo(float).eps
        
    def forward(self,z,z_dist):
        """
        :param z: the batchh of latent representations.
        :param z_dist: the batch of estimated cpds.
        :return: the mean log-likelihood(averagedalong the batch axis).
        """
        z_d=z.detach()
        z_dist=F.softmax(z_dist,dim=1)
        #Flatten out codes and distributions
        z_d=z_d.view(len(z_d),-1).contiguous()
        z_dist=z_dist.view(len(z_d),self.cpd_channels,-1).contiguous()
        #Log(regularized), pick the right ones
        z_dist=torch.clamp(z_dist,self.eps,1-self.eps)
        log_z_dist=z_dist
        index=torch.clamp(torch.unsqueeze(z_d,dim=1)*self.cpd_channels,min=0,max=(self.cpd_channels-1)).long()
        selected=torch.gather(log_z_dist,dim=1,index=index)
        #print("#################model/loss_func atr_loss selected_gather.shape:{}".format(selected.size()))
        selected=torch.squeeze(selected,dim=1)
        #Sum ans mean
        #S=torch.sum(selected,dim=-1)
        S=torch.mean(selected,dim=-1)
        #print("#################model/loss_func atr_loss z.shape:{}".format(z.size()))
        #print("#################model/loss_func atr_loss z_d.shape:{}".format(z_d.size()))
        #print("#################model/loss_func atr_loss z_dist.shape:{}".format(z_dist.size()))
        #print("#################model/loss_func atr_loss index.shape:{}".format(index.size()))
        #print("#################model/loss_func atr_loss selected.shape:{}".format(selected.size()))
        #print("#################model/loss_func atr_loss S.shape:{}".format(S.size()))
        #atr=-torch.mean(S)
        atr=torch.mean(S)
        return atr
        

class SMYLoss(torch.nn.Module):
    def __init__(self,cpd_channels,lam=1):
        """
        :param cpd_channels: number of bins in which the multinomial works.
        :param lam: weight of the autoregression loss.
        """
        super(SMYLoss,self).__init__()
        self.cpd_channels=cpd_channels
        self.lam=lam
        
        self.reconstruction_loss_fn=ReconstructionLoss()
        self.autoregression_loss_fn=AutoregressionLoss(cpd_channels)
        
    def forward(self,x_r,x,z,z_dist):
        """
        :param x: the batch of input samples.
        :param x_r: the batch of reconstructions.
        :param z: the batch of latent representations.
        :param z_dist: the batch of estimated cpds.
        :return: the loss of the model.
        """
        rec_loss=(self.reconstruction_loss_fn(x,x_r))
        atr_loss=self.autoregression_loss_fn(z,z_dist)
        tol_loss=rec_loss+self.lam*atr_loss
        return tol_loss,rec_loss,atr_loss
