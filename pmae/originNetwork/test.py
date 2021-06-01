import os
import cv2
import sys
import glob
import time
import argparse
import numpy as np
from collections import OrderedDict
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torch.utils.data.dataset as dataset
import torchvision.transforms as transforms
from torchvision import utils as vutils
from torch.autograd import Variable

from model.utils import DataLoader
from model.utils import anomaly_score_list,anomaly_score_list_inv,AUC,score_sum,psnr
from model.utils import ToFloatTensor3D
from model.utils import point_score
from model.encoder_backbone import EncoderBackbone
from model.loss_func import SMYLoss

parser=argparse.ArgumentParser(description="smy")
parser.add_argument("--gpus",nargs="+",type=str,help="gpus")
parser.add_argument("--batch_size",type=int,default=4,help="batch size for training")
parser.add_argument("--test_batch_size",type=int,default=1,help="batch size for test")
parser.add_argument("--h",type=int,default=256,help="height of input images")
parser.add_argument("--w",type=int,default=256,help="width of input images")
parser.add_argument("--c",type=int,default=3,help="channel of input images")
parser.add_argument("--t_length",type=int,default=5,help="length of the frame sequences")
parser.add_argument("--fdim",type=int,default=512,help="channel dimension of the features")
parser.add_argument("--alpha",type=float,default=0.6,help="weight for the anomality score")
parser.add_argument("--num_workers",type=int,default=2,help="number of workers for the train loader")
parser.add_argument("--num_workers_test",type=int,default=1,help="number of workers for the test loader")
parser.add_argument("--dataset_type",type=str,default="ped2",help="type of dataset: ped2, shanghai")
parser.add_argument("--dataset_path",type=str,default="./dataset/",help="directory of data")
parser.add_argument("--model_dir",type=str,default="./exp/ped2/log/ped2_epoch1_model.pth",help="directory of model")
parser.add_argument("--cpd_channels",type=int,default=100,help="channel dimension of the cpd")
parser.add_argument("--m_items_dir",type=str,help="directory of mem model")
parser.add_argument("--th",type=float,default=0.01,help="threshold for test updating")
args=parser.parse_args()

torch.manual_seed(2020)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if args.gpus is None:
    gpus="0"
    os.environ["CUDA_VISIBLE_DEVICES"]=gpus
else:
    gpus=""
    for i in range(len(args.gpus())):
        gpus=gpus+args.gpus[i]+","
    os.environ["CUDA_VISIBLE_DEVICES"]=gpus[:-1]
torch.backends.cudnn.enabled=True
print("###############test.py, set done###############")

if args.dataset_type=="ped2":
    test_folder=args.dataset_path+args.dataset_type+"/Test"
else:
    test_folder=args.dataset_path+args.dataset_type+"/testing/frames"
test_dataset=DataLoader(args.dataset_type,test_folder,transforms.Compose([
                        #transforms.ToTensor(),#(h,w,c)->(c,h,w)
                        #RemoveBackground(threshold=128),
                        ToFloatTensor3D(normalize=True),
                        ]),resize_height=args.h,resize_width=args.w,time_step=args.t_length-1)
test_batch=data.DataLoader(test_dataset,batch_size=args.test_batch_size,
                            shuffle=False,num_workers=args.num_workers_test,drop_last=False)
print("###############test.py, dataload done###############")

model=torch.load(args.model_dir)
m_items=torch.load(args.m_items_dir)
#loss_func_mse=nn.MSELoss(reduction="none")
loss_func=SMYLoss(cpd_channels=args.cpd_channels)
model.cuda()
print("###############test.py, model load done###############")

labels=np.load("./data/frame_labels_"+args.dataset_type+".npy")
if args.dataset_type=="shanghai":
    labels=np.expand_dims(labels,0)
print("###############test.py, label load done###############")

videos=OrderedDict()
videos_list=sorted(glob.glob(os.path.join(test_folder,"*")))
for video in videos_list:
    video_name=video.split("/")[-1]
    videos[video_name]={}
    videos[video_name]["path"]=video
    if args.dataset_type=="ped2":
        videos[video_name]["frame"]=glob.glob(os.path.join(video,"*.tif"))
    else:
        videos[video_name]["frame"]=glob.glob(os.path.join(video,"*.*"))
    videos[video_name]["frame"].sort()
    videos[video_name]["length"]=len(videos[video_name]["frame"])
    
labels_list=[]
label_length=0
psnr_list={}
atr_list={}
feature_distance_list={}
for video in sorted(videos_list):
    video_name=video.split("/")[-1]
    labels_list=np.append(labels_list,labels[0][args.t_length-1+label_length:videos[video_name]["length"]+label_length])
    label_length+=videos[video_name]["length"]
    psnr_list[video_name]=[]
    atr_list[video_name]=[]
    feature_distance_list[video_name]=[]

label_length=0
video_num=0
label_length+=videos[videos_list[video_num].split("/")[-1]]["length"]
m_items_test=m_items.clone()
model.eval()

for k,(imgs) in enumerate(test_batch):
    if k==label_length-(args.t_length-1)*(video_num+1):
        video_num+=1
        label_length+=videos[videos_list[video_num].split("/")[-1]]["length"]
        
    imgs=Variable(imgs).cuda()
    start_time=time.time()
    outputs,z,z_dist,m_items_test,compactness_loss,_,_,_,score_query,score_memory=model.forward(imgs[:,:,0:args.t_length-1],m_items_test,False)
    gap_time=time.time()-start_time
    print("frame:{} time:{}".format(k,gap_time))
    #model test return:  x_r,z,z_dist,mems,compactness_loss,query,top1_key,key_indice,score_query,score_memory
    #print("##########################test.py pre_target_img.size():{}".format(imgs[:,:,args.t_length-1:].shape))
    #print("##########################test.py pre_output_img.size():{}".format(outputs.shape))
    #mse_imgs=torch.mean(loss_func_mse((outputs+1)/2,(imgs[:,:,args.t_length-1:]+1)/2)).item()#rec_score
    outputs_pic=outputs[0].clone().detach()
    outputs_pic=outputs_pic.to(torch.device("cpu"))
    outputs_pic=outputs_pic.squeeze()
    outputs_pic=outputs_pic.add_(1.0).mul_(127.5).clamp_(0,255).permute(1,2,0).type(torch.uint8).numpy()
    outputs_pic=cv2.cvtColor(outputs_pic,cv2.COLOR_RGB2BGR)
    cv2.imwrite("./pic/{}.png".format(k),outputs_pic)
    #print("outputs_pic.size():{}".format(outputs_pic.size()))
    gt_pic=imgs[:,:,args.t_length-1:].clone().detach()
    gt_pic=gt_pic.to(torch.device("cpu"))
    gt_pic=gt_pic.squeeze()
    gt_pic=gt_pic.add_(1).mul_(127.5).clamp_(0,255).permute(1,2,0).type(torch.uint8).numpy()
    gt_pic=cv2.cvtColor(gt_pic,cv2.COLOR_RGB2BGR)
    #vutils.save_image(outputs_pic,"./pics/{}.png".format(k))
    #vutils.save_image(gt_pic,"./pics/{}_gt.png".format(k))
    cv2.imwrite("./pic/{}_gt.png".format(k),gt_pic)
    loss_pixel,rec_loss,atr_loss=loss_func(outputs,imgs[:,:,args.t_length-1:],z,z_dist)
    mse_imgs=(rec_loss.item()+1)/4
    mse_feas=compactness_loss.item()
    
    point_sc=point_score(outputs,imgs[:,:,args.t_length-1:])
    if point_sc<args.th:
        query=F.normalize(z_dist,dim=1)
        m_items_test=model.memory.update(query,m_items_test,False)
    
    atr_list[videos_list[video_num].split("/")[-1]].append(atr_loss.item())
    psnr_list[videos_list[video_num].split("/")[-1]].append(psnr(mse_imgs))
    feature_distance_list[videos_list[video_num].split("/")[-1]].append(mse_feas)
    #print("#####################test.py mse_img.size():{} psnr(mse_img):{} atr_loss.item():{}".format(mse_imgs,psnr(mse_imgs),atr_loss.item()))

anomaly_score_total_list=[]
for video in sorted(videos_list):
    video_name=video.split("/")[-1]
    ##################anomaly_score_single_list=(anomaly_score_list(psnr_list[video_name]))#rec_score
    #anomaly_score_single_rec_list=(psnr_list[video_name])
    #anomaly_score_single_atr_list=(atr_list[video_name])
    #anomaly_score_single_mem_list=(feature_distance_list[video_name])
    anomaly_score_single_rec_list=anomaly_score_list(psnr_list[video_name])
    anomaly_score_single_atr_list=anomaly_score_list(atr_list[video_name])
    anomaly_score_single_mem_list=anomaly_score_list_inv(feature_distance_list[video_name])
    anomaly_score_single_list=(anomaly_score_list(psnr_list[video_name]))
    #anomaly_score_single_list=score_sum(anomaly_score_single_rec_list,
    #                                    anomaly_score_single_atr_list,
    #                                    anomaly_score_single_mem_list,
    #                                    args.alpha)
    """
    anomaly_score_single_list=score_sum(anomaly_score_list(psnr_list[video_name]),
                                        anomaly_score_list_inv(feature_distance_list[video_name]),
                                        args.alpha)
    """
    result_folder_path=os.path.join("/".join(args.model_dir.split("/")[:-1]),args.model_dir.split("_")[1])
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)
    for i in range(len(anomaly_score_single_list)):
        result_file_path=os.path.join(result_folder_path,"{}_results.txt".format(videos[video_name]["frame"][i].split("/")[-2]))
        with open(result_file_path,'a') as f:
            f.write("frame:{} total_score:{:.6f} rec_score:{:.6f} atr_score:{:.6f} mem_score:{:.6f}\n".format(videos[video_name]["frame"][i].split("/")[-1].split(".")[0],
            float(anomaly_score_single_list[i]),
            float(anomaly_score_single_rec_list[i]),
            float(anomaly_score_single_atr_list[i]),
            float(anomaly_score_single_mem_list[i])
            ))
    anomaly_score_total_list+=anomaly_score_single_list
    
#anomaly_score_total_list=np.asarray(anomaly_score_total_list)
#accuracy=AUC(anomaly_score_total_list,np.expand_dims(1-labels_list,0))
#with open(os.path.join("/".join(args.model_dir.split("/")[:-1]),"result.txt"),"a") as f:
#    f.write("{} {:.6f}\n".format(args.model_dir.split("_")[1],accuracy))
#print("the result of ",args.dataset_type)
#print("AUC: ",accuracy*100,"%")
print("######################test.py test done##############")
