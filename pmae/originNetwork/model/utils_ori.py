import os
import cv2
import glob
import math
import numpy as np
from collections import OrderedDict
from sklearn.metrics import roc_auc_score

import torch
import torch.utils.data as data

rng=np.random.RandomState(2020)

class ToFloatTensor3D(object):
    def __init__(self,normalize=True):
        self._normalize=normalize
        
    def __call__(self,sample):
        X=sample
        #print("###############smy model/utils/transpose before img.shape:{}".format((X.shape)))
        X=X.transpose(3,0,1,2)
        #print("###############smy model/utils/transpose after img.shape:{}".format((X.shape)))#(b,t,h,w,c)->(b,c,t,h,w)
        return torch.from_numpy(X)
        

def np_load_frame(filename,resize_height,resize_width):
    """
    load image and convert to numpy.
    color channels are BGR.
    color space is normalized from [0,255] to [-1,1].
    
    :param filename: the full path of image
    :param resize_height: resized height
    :param resize_width: resized width
    :return: numpy.ndarray
    """
    image_decoded=cv2.imread(filename)
    image_resized=cv2.resize(image_decoded,(resize_width,resize_height))
    image_resized=image_resized.astype(dtype=np.float32)
    image_resized=(image_resized/127.5)-1.0
    return image_resized

class DataLoader(data.Dataset):
    def __init__(self,dataset_type,video_folder,transform,resize_height,resize_width,time_step=4,num_pred=1):
        self.dataset_type=dataset_type
        self.dir=video_folder
        self.transform=transform
        self.videos=OrderedDict()
        self._resize_height=resize_height
        self._resize_width=resize_width
        self._time_step=time_step
        self._num_pred=num_pred
        self.setup()
        self.samples=self.get_all_samples()
        
    def setup(self):
        videos=[d for d in glob.glob(os.path.join(self.dir,"**")) if os.path.isdir(d) and "gt" not in os.path.basename(d)]
        for video in sorted(videos):
            video_name=video.split("/")[-1]
            self.videos[video_name]={}
            self.videos[video_name]["path"]=video
            if(self.dataset_type=="ped2"):
                self.videos[video_name]["frame"]=glob.glob(os.path.join(video,"*.tif"))
            else:
                self.videos[video_name]["frame"]=glob.glob(os.path.join(video,"*.*"))
            self.videos[video_name]["frame"].sort()
            self.videos[video_name]["length"]=len(self.videos[video_name]["frame"])
            
    def get_all_samples(self):
        frames=[]
        videos=glob.glob(os.path.join(self.dir,"*"))
        for video in sorted(videos):
            video_name=video.split("/")[-1]
            for i in range(len(self.videos[video_name]["frame"])-self._time_step):
                frames.append(self.videos[video_name]["frame"][i])
        return frames
        
    def __getitem__(self,index):
        video_name=self.samples[index].split("/")[-2]
        frame_name=int(self.samples[index].split("/")[-1].split(".")[-2])
        batch=[]
        for i in range(self._time_step+self._num_pred):
            if self.dataset_type=="ped2":
                image=np_load_frame(self.videos[video_name]["frame"][frame_name-1+i],self._resize_height,self._resize_width)
            else:
                image=np_load_frame(self.videos[video_name]["frame"][frame_name+i],self._resize_height,self._resize_width)
            batch.append((image))
        #print("###############utils.py dataloader len(batch):{}".format((batch[0].shape)))
        if self.transform is not None:
            batch=self.transform(np.asarray(batch))
        #return np.concatenate(batch,axis=0)
        return batch
        
    def __len__(self):
        return len(self.samples)

def point_score(outputs,imgs):
    loss_func_mse=torch.nn.MSELoss(reduction="none")
    error=loss_func_mse((outputs+1)/2,(imgs+1)/2)
    normal=1-torch.exp(-error)
    score=(torch.sum(normal*loss_func_mse((outputs+1)/2,(imgs+1)/2))/torch.sum(normal)).item()
    return score

def psnr(mse):
    return 10*math.log10(1/mse)

def anomaly_score(psnr,max_psnr,min_psnr):
    if max_psnr==min_psnr:
        return 0
    return ((psnr-min_psnr)/(max_psnr-min_psnr))
    
def anomaly_score_inv(psnr,max_psnr,min_psnr):
    if max_psnr==min_psnr:
        return 1
    return (1.0-((psnr-min_psnr)/(max_psnr-min_psnr)))
    
def anomaly_score_list(psnr_list):
    anomaly_score_list=list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score(psnr_list[i],np.max(psnr_list),np.min(psnr_list)))
    return anomaly_score_list
    
def anomaly_score_list_inv(psnr_list):
    anomaly_score_list=list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score_inv(psnr_list[i],np.max(psnr_list),np.min(psnr_list)))
    return anomaly_score_list
    
def AUC(anomal_scores,labels):
    frame_auc=roc_auc_score(y_true=np.squeeze(labels,axis=0),y_score=np.squeeze(anomal_scores))
    return frame_auc
    
def score_sum(list1,list2,list3,alpha):
    list_result=[]
    for i in range(len(list1)):
        list_result.append((alpha*list1[i]+(1-alpha)*list3[i]))
    return list_result
