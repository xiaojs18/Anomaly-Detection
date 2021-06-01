import os
import sys
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torch.utils.data.dataset as dataset
import torchvision.transforms as transforms
from torch.autograd import Variable

from model.utils import DataLoader
from model.utils import ToFloatTensor3D
from model.encoder_backbone import EncoderBackbone
from model.loss_func import SMYLoss

parser=argparse.ArgumentParser(description="smy")
parser.add_argument("--gpus",nargs="+",type=str,help="gpus")
parser.add_argument("--batch_size",type=int,default=4,help="batch size for training")
parser.add_argument("--test_batch_size",type=int,default=1,help="batch size for test")
parser.add_argument("--epochs",type=int,default=60,help="number of epochs for training")
parser.add_argument("--h",type=int,default=256,help="height of input images")
parser.add_argument("--w",type=int,default=256,help="width of input images")
parser.add_argument("--c",type=int,default=3,help="channel of input images")
parser.add_argument("--lr",type=float,default=2e-4,help="initial learning rate")
parser.add_argument("--t_length",type=int,default=5,help="length of the frame sequences")
parser.add_argument("--fdim",type=int,default=512,help="channel dimension of the features")
parser.add_argument("--num_workers",type=int,default=2,help="number of workers for the train loader")
parser.add_argument("--num_workers_test",type=int,default=1,help="number of workers for the test loader")
parser.add_argument("--dataset_type",type=str,default="ped2",help="type of dataset: ped2, shanghai")
parser.add_argument("--dataset_path",type=str,default="./dataset/",help="directory of data")
parser.add_argument("--exp_dir",type=str,default="log",help="directory of log")
parser.add_argument("--cpd_channels",type=int,default=100,help="channel dimension of the cpd")
parser.add_argument("--msize",type=int,default=10,help="number of the memory items")
parser.add_argument("--mdim",type=int,default=512,help="dimension of the memory items")
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
print("###############train.py, set done###############")

if args.dataset_type=="ped2":
    train_folder=args.dataset_path+args.dataset_type+"/Train"
else:
    train_folder=args.dataset_path+args.dataset_type+"/training/frames"
train_dataset=DataLoader(args.dataset_type,train_folder,transforms.Compose([
                        #transforms.ToTensor(),#(h,w,c)->(c,h,w)
                        #RemoveBackground(threshold=128),
                        ToFloatTensor3D(normalize=True),
                        ]),resize_height=args.h,resize_width=args.w,time_step=args.t_length-1)
train_batch=data.DataLoader(train_dataset,batch_size=args.batch_size,
                            shuffle=True,num_workers=args.num_workers,drop_last=True)
print("###############train.py, data load done. dataset_type:{},t_length:{}###############".format(args.dataset_type,args.t_length))

model=EncoderBackbone(n_channel=args.c,t_length=args.t_length,feature_dim=args.fdim,cpd_channels=args.cpd_channels,memory_size=args.msize,memory_dim=args.mdim)
params_encoder=list(model.encoder.parameters())
params_decoder=list(model.decoder.parameters())
params_estimator=list(model.estimator.parameters())
params=params_encoder+params_decoder+params_estimator
optimizer=torch.optim.Adam(params,lr=args.lr)
scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)
#loss_func_mse=nn.MSELoss(reduction="none")
loss_func=SMYLoss(cpd_channels=args.cpd_channels)
model.cuda()
print("###############train.py, model build done###############")

'''
param_show=list(model.parameters())
for i in params:
    print("struct:",str(list(i.size())))
'''

m_items=F.normalize(torch.rand((args.msize,args.mdim),dtype=torch.float),dim=1).cuda()

log_dir=os.path.join("./exp",args.dataset_type,args.exp_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
orig_stdout=sys.stdout
f=open(os.path.join(log_dir,"log.txt"),"w")
sys.stdout=f

for epoch in range(args.epochs):
    print("############train.py epoch:{}".format(epoch))
    label_list=[]
    model.train()
    start=time.time()
    for j,(imgs) in enumerate(train_batch):
        #print("#############train.py train_batch_j:{}".format(j))
        imgs=Variable(imgs).cuda()
        outputs,z,z_dist,m_items,compactness_loss,separateness_loss,score_query,score_memory=model.forward(imgs[:,:,0:args.t_length-1],m_items,True)
        #print("###########################train.py z_dist.shape:{}".format(z_dist.shape))
        optimizer.zero_grad()
        #print("##########################train.py img.size():{}".format(imgs.shape))
        #print("##########################train.py pre_target_img.size():{}".format(imgs[:,:,args.t_length-1:].shape))
        #print("##########################train.py pre_output_img.size():{}".format(outputs.shape))
        loss_pixel,rec_loss,atr_loss=loss_func(outputs,imgs[:,:,args.t_length-1:],z,z_dist)
        loss=loss_pixel+compactness_loss+separateness_loss
        loss.backward(retain_graph=True)
        optimizer.step()
    scheduler.step()
    epoch_train_time=time.time()-start
    
    print("-------------------------------------")
    print("Epoch:",epoch+1)
    print("Loss:{:.6f} rec_loss:{:.6f} atr_loss:{:.6f} compactness_loss:{:.6f} separateness_loss:{:.6f} train_time:{:.6f}".format(loss.item(),rec_loss.item(),atr_loss.item(),compactness_loss.item(),separateness_loss.item(),epoch_train_time))
    torch.save(model,os.path.join(log_dir,"{}_epoch{}_model.pth".format(args.dataset_type,epoch+1)))
    torch.save(m_items,os.path.join(log_dir,"{}_epoch{}_mem.pth".format(args.dataset_type,epoch+1)))

sys.stdout=orig_stdout
f.close()    
print("######################train.py train done##############")
