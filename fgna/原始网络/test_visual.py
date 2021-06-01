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
    #test_visual_path='test_visual'
    from sp_layer import Mil_Attention, Bag_Pooling
    from keras.regularizers import l2
    from keras.models import Model
    #from keras import backend as K
    from matplotlib import pyplot as plt
    from sp_loss import bag_accuracy, bag_loss
    #if not os.path.exists(test_result_path):
    #    os.makedirs(test_result_path)
    model=load_model('./train/model_snapshot_epoch{:04d}_loss{}.h5'.format(epoch,loss),custom_objects={'Mil_Attention' : Mil_Attention, 'Bag_Pooling': Bag_Pooling,'bag_loss' : bag_loss, 'bag_accuracy':bag_accuracy})
    model.summary()
    flatten_layer_model = Model(inputs=model.input,outputs=model.get_layer('flatten_1').output)
    multiply_layer_model = Model(inputs=model.input,outputs=model.get_layer('multiply_1').output)
    #fc2_layer_model = Model(inputs=model.input,outputs=model.get_layer('fc2').output)
    alpha_layer_model = Model(inputs=model.input,outputs=model.get_layer('alpha').output)
    
    #pain_model()
    for video_path in tqdm(os.listdir(test_video_path)):
        video=os.listdir(os.path.join(test_video_path,video_path))
        video.sort()
        print("######process {}".format(video_path))
        imagedata=[]
        #img_fc2=[]
        img_alpha=[]
        #img_mil=[]
        img_flatten=[]
        img_multiply=[]
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

            #---------------flatten--------------------
            flatten_predict=flatten_layer_model.predict(imagedata)
            flatten_predict=np.squeeze(flatten_predict)
            flatten_predict=flatten_predict+abs(np.min(flatten_predict))
            flatten_predict=flatten_predict/np.max(flatten_predict)
            #img_flatten.append(flatten_predict*255)
            flatten_predict=np.reshape(flatten_predict*255,(38,38))
            plt.imshow(flatten_predict,cmap='gray')
            plt.axis('off')
            plt.savefig('./pic_visual/pic_flatten/{}.png'.format(video_name[:-4]))
            
            
            #---------------fc2--------------------
            #fc2_predict=fc2_layer_model.predict(imagedata)
            #fc2_predict=np.squeeze(fc2_predict)
            ##model_predict=model_predict+np.min(model_predict)*np.ones(model_predict.shape)
            #fc2_predict=fc2_predict+abs(np.min(fc2_predict))
            #fc2_predict=fc2_predict/np.max(fc2_predict)
            #img_fc2.append(fc2_predict*255)
            #print(frame_num,fc2_predict.shape)
            
            #layer_1 = K.function([model.layers[0].input], [model.layers[14].output])
            #f1 = layer_1([imagedata])[0]
            
            #---------------alpha--------------------
            alpha_predict=alpha_layer_model.predict(imagedata)
            alpha_predict=np.squeeze(alpha_predict)
            #print(frame_num,alpha_predict)
            alpha_predict=alpha_predict+abs(np.min(alpha_predict))
            alpha_predict=alpha_predict/np.max(alpha_predict)
            alpha_predict=np.reshape(alpha_predict*255,(38,38))
            plt.imshow(alpha_predict,cmap='gray')
            plt.axis('off')
            plt.savefig('./pic_visual/pic_alpha/{}.png'.format(video_name[:-4]))
            #img_alpha.append(alpha_predict*255)
            print(frame_num,alpha_predict.shape,alpha_predict)
            
            #---------------multiply--------------------
            #mil_predict=fc2_predict*alpha_predict
            #img_mil.append(mil_predict*255)
            #print(frame_num,mil_predict.shape)
            multiply_predict=multiply_layer_model.predict(imagedata)
            multiply_predict=np.squeeze(multiply_predict)
            multiply_predict=multiply_predict+abs(np.min(multiply_predict))
            multiply_predict=multiply_predict/np.max(multiply_predict)
            multiply_predict=np.reshape(multiply_predict*255,(38,38))
            plt.imshow(multiply_predict,cmap='gray')
            plt.axis('off')
            plt.savefig('./pic_visual/pic_multiply/{}.png'.format(video_name[:-4]))
            #img_multiply_1.append(multiply_1*255)
            #print(frame_num,multiply_1_predict.shape,np.min(multiply_1_predict),np.max(multiply_1_predict))
            
            
            ##cv2.imwrite('./pic/{}.png'.format(frame_num),model_predict*255)
            ##with open(os.path.join('./paper_inspect','{}_predict.txt'.format(video_path)),'a') as f:
            ##    f.write('{:04d} predict {}\n'.format(frame_num-8,model_predict))
        #cv2.imwrite('./fusion.png',np.array(img_fusion))
        #cv2.imwrite('./fc2.png',np.array(img_fc2))
        #cv2.imwrite('./alpha.png',np.array(img_alpha))
        #cv2.imwrite('./mil.png',np.array(img_mil))
        print("save done!")
        
    
#test()
test_pic()
#source activate smy-ab2
#python test_visual.py --test_video_path='./test_visual' --epoch=20 --loss=1.020295
