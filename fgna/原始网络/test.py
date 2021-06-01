from keras.models import load_model
import numpy as np
import argparse
import os
import cv2
from tqdm import tqdm
import yaml
from scipy.misc import imresize

parser=argparse.ArgumentParser(description='test param')
parser.add_argument('--test_video_path',type=str)
parser.add_argument('--epoch',type=int)
parser.add_argument('--loss',type=float)
parser.add_argument('--save_path',type=str)
args=parser.parse_args()

test_video_path=args.test_video_path
epoch=args.epoch 
loss=args.loss
test_result_path=args.save_path

with open('config.yml','r') as yamlfile:
    cfg=yaml.load(yamlfile)

img_W=cfg['img_W']
img_H=cfg['img_H']
time_length=cfg['time_length']
weight_decay=cfg['init_lr']
useGated=cfg['useGated']
img_size=(img_W,img_H)



def test_pic():
    #test_result_path='test'
    from sp_layer import Mil_Attention, Bag_Pooling
    from keras.regularizers import l2
    from sp_loss import bag_accuracy, bag_loss
    if not os.path.exists(test_result_path):
        os.makedirs(test_result_path)
    model=load_model('./train/model_snapshot_epoch{:04d}_loss{:06f}.h5'.format(epoch,loss),custom_objects={'Mil_Attention' : Mil_Attention, 'Bag_Pooling': Bag_Pooling,'bag_loss' : bag_loss, 'bag_accuracy':bag_accuracy})
    model.summary()
    for video_path in tqdm(os.listdir(test_video_path)):
        video=os.listdir(os.path.join(test_video_path,video_path))
        video.sort()
        print("######process {}".format(video_path))
        imagedata=[]
        frame_num=1
        for video_name in video:
            if frame_num==8:
                frame=cv2.imread(os.path.join(test_video_path,video_path,video_name))
                #print(os.path.join(test_video_path,video_path,video_name),frame.shape)
                frame=imresize(frame,img_size)
                gray=0.2989*frame[:,:,0]+0.5870*frame[:,:,1]+0.1140*frame[:,:,2]
                gray=(gray-gray.mean()+gray.std())/(2*gray.std())
                gray=np.clip(gray,0,1)
                imagedata.append(gray)
                frame_num+=1
                imagedata=np.expand_dims(imagedata,axis=-1)
                imagedata=np.expand_dims(imagedata,axis=0)
            elif frame_num<8:
                frame=cv2.imread(os.path.join(test_video_path,video_path,video_name))
                #print(os.path.join(test_video_path,video_path,video_name),frame.shape)
                frame=imresize(frame,img_size)
                gray=0.2989*frame[:,:,0]+0.5870*frame[:,:,1]+0.1140*frame[:,:,2]
                gray=(gray-gray.mean()+gray.std())/(2*gray.std())
                gray=np.clip(gray,0,1)
                imagedata.append(gray)
                frame_num+=1
                continue
            elif frame_num>8:
                frame=cv2.imread(os.path.join(test_video_path,video_path,video_name))
                #print(os.path.join(test_video_path,video_path,video_name),frame.shape)
                frame=imresize(frame,img_size)
                gray=0.2989*frame[:,:,0]+0.5870*frame[:,:,1]+0.1140*frame[:,:,2]
                gray=(gray-gray.mean()+gray.std())/(2*gray.std())
                gray=np.clip(gray,0,1)
                gray=np.expand_dims(gray,axis=0)
                imagedata=np.squeeze(imagedata)
                imagedata=imagedata[1:,:,:]
                imagedata=np.row_stack((imagedata,gray))
                imagedata=np.expand_dims(imagedata,axis=0)
                imagedata=np.expand_dims(imagedata,axis=-1)
                frame_num+=1

            model_predict=model.predict(imagedata)
            model_predict=np.squeeze(model_predict)
            ###save_img(video_name,frame_num,imagedata,model_predict)
            with open(os.path.join(test_result_path,'{}_predict.txt'.format(video_path)),'a') as f:
                f.write('{:04d} predict {}\n'.format(frame_num-8,model_predict))
            
        with open(os.path.join(test_result_path,'{}_predict.txt'.format(video_path)),'a') as f:
            for i in range(time_length-1):
                f.write('{:04d} predict {}\n'.format(frame_num-7,model_predict))
                frame_num+=1
        print("######process {} done, total frame:{}".format(video_path,frame_num-7))
    
test_pic()
#python test.py --test_video_path='./testset_33' --epoch=1 --loss=0.076799  --save_path='./test_1'
