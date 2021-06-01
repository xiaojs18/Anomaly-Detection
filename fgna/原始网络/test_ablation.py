from keras.models import load_model
import numpy as np
import argparse
import os
import cv2
from tqdm import tqdm
import yaml
from scipy.misc import imresize
from keras.models import Model
from keras.layers.convolutional import Conv2D,Conv2DTranspose
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Activation
from keras.layers import Input,Flatten,Dense, Dropout, multiply, MaxPooling2D,Conv3D
from keras.regularizers import l2
#from keras.layers import Conv3D,ConvLSTM2D,Conv3DTranspose, MaxPooling2D
import yaml
from keras import optimizers
from sp_loss import bag_accuracy, bag_loss
from sp_layer import Mil_Attention, Bag_Pooling

parser=argparse.ArgumentParser(description='test param')
parser.add_argument('--test_video_path',type=str)
parser.add_argument('--epoch',type=int)
parser.add_argument('--loss',type=float)
args=parser.parse_args()

test_video_path=args.test_video_path
epoch=args.epoch 
loss=args.loss 

with open('config.yml','r') as yamlfile:
    cfg=yaml.load(yamlfile)

img_W=cfg['img_W']
img_H=cfg['img_H']
time_length=cfg['time_length']
weight_decay=cfg['init_lr']
useGated=cfg['useGated']
img_size=(img_W,img_H)

def load_model_withatt():
    input_tensor=Input(shape=(time_length,img_W,img_H,1))
    conv1=TimeDistributed(Conv2D(128,kernel_size=(5,5),padding='same',strides=(3,3),name='conv1'),input_shape=(time_length,img_W,img_H,1))(input_tensor)
    conv1=TimeDistributed(BatchNormalization())(conv1)
    conv1=TimeDistributed(Activation('relu'))(conv1)
    
    conv2=TimeDistributed(Conv2D(64,kernel_size=(3,3),padding='same',strides=(2,2),name='conv2'))(conv1)
    conv2=TimeDistributed(BatchNormalization())(conv2)
    conv2=TimeDistributed(Activation('relu'))(conv2)
    
    convlstm1=ConvLSTM2D(64,kernel_size=(3,3),padding='same',return_sequences=True,name='convlstm1')(conv2)
    convlstm2=ConvLSTM2D(32,kernel_size=(3,3),padding='same',return_sequences=True,name='convlstm2')(convlstm1)
    
    fusion=Conv3D(1, kernel_size=(time_length,1,1), strides=1,name='fusion')(convlstm2)
    x = Flatten()(fusion)
    alpha = Mil_Attention(L_dim=128, output_dim=1444, kernel_regularizer=l2(weight_decay), name='alpha', use_gated=useGated)(x)
    x_mul = multiply([alpha, x])
    fc1 = Dense(512, activation='relu',kernel_regularizer=l2(weight_decay), name='fc1')(x_mul)
    fc1 = Dropout(0.5)(fc1)
    fc2 = Dense(512, activation='relu',kernel_regularizer=l2(weight_decay), name='fc2')(fc1)
    fc2 = Dropout(0.5)(fc2)
    
    out = Bag_Pooling(output_dim=1, name='FC1_sigmoid')(fc2)
    return Model(inputs=[input_tensor],outputs=[out])

def load_model_withoutatt():
    input_tensor=Input(shape=(time_length,img_W,img_H,1))
    conv1=TimeDistributed(Conv2D(128,kernel_size=(5,5),padding='same',strides=(3,3),name='conv1'),input_shape=(time_length,img_W,img_H,1))(input_tensor)
    conv1=TimeDistributed(BatchNormalization())(conv1)
    conv1=TimeDistributed(Activation('relu'))(conv1)
    
    conv2=TimeDistributed(Conv2D(64,kernel_size=(3,3),padding='same',strides=(2,2),name='conv2'))(conv1)
    conv2=TimeDistributed(BatchNormalization())(conv2)
    conv2=TimeDistributed(Activation('relu'))(conv2)
    
    convlstm1=ConvLSTM2D(64,kernel_size=(3,3),padding='same',return_sequences=True,name='convlstm1')(conv2)
    convlstm2=ConvLSTM2D(32,kernel_size=(3,3),padding='same',return_sequences=True,name='convlstm2')(convlstm1)
    
    fusion=Conv3D(1, kernel_size=(time_length,1,1), strides=1,name='fusion')(convlstm2)
    x = Flatten()(fusion)
    fc1 = Dense(512, activation='relu',kernel_regularizer=l2(weight_decay), name='fc1')(x)
    fc1 = Dropout(0.5)(fc1)
    fc2 = Dense(512, activation='relu',kernel_regularizer=l2(weight_decay), name='fc2')(fc1)
    fc2 = Dropout(0.5)(fc2)
    
    out = Bag_Pooling(output_dim=1, name='FC1_sigmoid')(fc2)
    return Model(inputs=[input_tensor],outputs=[out])
   
def pain_model(model):
    from keras.utils import plot_model
    plot_model(model,to_file='model_test.png',show_shapes=True)
    
def save_img(video_name,frame_num,imagedata,model_predict):
    input_folder='./process_img/{}/input_img/{}'.format(video_name,frame_num)
    output_folder='./process_img/{}/output_img/{}'.format(video_name,frame_num)
    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for i in range(imagedata.shape[1]):
        cv2.imwrite(os.path.join(input_folder,'{}.png'.format(i)),imagedata[0][i]*255)
    for i in range(model_predict.shape[1]):
        cv2.imwrite(os.path.join(output_folder,'{}.png'.format(i)),model_predict[0][i]*255)

def test_pic():
    test_result_path='test'
    from sp_layer import Mil_Attention, Bag_Pooling
    from keras.regularizers import l2
    from keras.models import Model
    #from keras import backend as K
    from matplotlib import pyplot as plt
    from sp_loss import bag_accuracy, bag_loss
    #if not os.path.exists(test_result_path):
    #    os.makedirs(test_result_path)
    
    
    
    model_withatt=load_model_withatt()
    model_withatt.summary()
    model_withatt.load_weights('./train/model_snapshot_epoch{:04d}_loss{}.h5'.format(epoch,loss),by_name=True)
    
    model_withoutatt=load_model_withoutatt()
    #model_withoutatt.summary()
    model_withoutatt.load_weights('./train/model_snapshot_epoch{:04d}_loss{}.h5'.format(epoch,loss),by_name=True)
    
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

            
            assert frame_num<20
            model_predict_withatt=model_withatt.predict(imagedata)
            model_predict_withatt=np.squeeze(model_predict_withatt)
            with open(os.path.join(test_result_path+'_withatt','{}_predict.txt'.format(video_path)),'a') as f:
                f.write('{:04d} predict {}\n'.format(frame_num-8,model_predict_withatt))
            
            model_predict_withoutatt=model_withoutatt.predict(imagedata)
            model_predict_withoutatt=np.squeeze(model_predict_withoutatt)
            with open(os.path.join(test_result_path+'_withoutatt','{}_predict.txt'.format(video_path)),'a') as f:
                f.write('{:04d} predict {}\n'.format(frame_num-8,model_predict_withoutatt))
            
        
        
        with open(os.path.join(test_result_path+'_withatt','{}_predict.txt'.format(video_path)),'a') as f:
            for i in range(time_length-1):
                f.write('{:04d} predict {}\n'.format(frame_num-7,model_predict_withatt))
                frame_num+=1
        
        frame_num-=7
        
        with open(os.path.join(test_result_path+'_withoutatt','{}_predict.txt'.format(video_path)),'a') as f:
            for i in range(time_length-1):
                f.write('{:04d} predict {}\n'.format(frame_num-7,model_predict_withoutatt))
                frame_num+=1
        print("######process {} done, total frame:{}".format(video_path,frame_num-7))
        print("save done!")
        
    
#test()
test_pic()
#python test_ablation.py --test_video_path='./testset_33' --epoch=20 --loss=1.020295
