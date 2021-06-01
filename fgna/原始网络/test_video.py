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
    

def test():
    from sp_layer import Mil_Attention, Bag_Pooling
    from keras.regularizers import l2
    from sp_loss import bag_accuracy, bag_loss
    test_result_path='test'
    if not os.path.exists(test_result_path):
        os.makedirs(test_result_path)
    model=load_model('./train/model_snapshot_epoch{:04d}_loss{}.h5'.format(epoch,loss),custom_objects={'Mil_Attention' : Mil_Attention, 'Bag_Pooling': Bag_Pooling,'bag_loss' : bag_loss, 'bag_accuracy':bag_accuracy})
    for video_path in tqdm(os.listdir(test_video_path)):
        video=os.path.join(test_video_path,video_path)
        video_name=video.split('/')[-1].split('.')[0]
        print("######process {}".format(video))
        imagedata=[]
        video_data=cv2.VideoCapture(video)
        rval,frame_pro=video_data.read()
        frame_num=1
        while rval:
            frame=frame_pro
            if frame_num==1:
                for frame_num_i in range(time_length):
                    frame=imresize(frame,img_size)
                    gray=0.2989*frame[:,:,0]+0.5870*frame[:,:,1]+0.1140*frame[:,:,2]
                    gray=(gray-gray.mean()+gray.std())/(2*gray.std())
                    gray=np.clip(gray,0,1)
                    imagedata.append(gray)
                    rval,frame=video_data.read()
                imagedata=np.expand_dims(imagedata,axis=-1)
                imagedata=np.expand_dims(imagedata,axis=0)
            else:
                imagedata=np.squeeze(imagedata)
                imagedata=imagedata[1:,:,:]
                frame=imresize(frame,img_size)
                gray=0.2989*frame[:,:,0]+0.5870*frame[:,:,1]+0.1140*frame[:,:,2]
                gray=(gray-gray.mean()+gray.std())/(2*gray.std())
                gray=np.clip(gray,0,1)
                gray=np.expand_dims(gray,axis=0)
                imagedata=np.row_stack((imagedata,gray))
                imagedata=np.expand_dims(imagedata,axis=0)
                imagedata=np.expand_dims(imagedata,axis=-1)
                           
            model_predict=model.predict(imagedata)
            model_predict=np.squeeze(model_predict)
            
            if model_predict<0.5:
                color_line=(0,255,0)
            else:
                color_line=(255,0,0)
            frame_pro=cv2.putText(frame_pro,"{:.3f}".format(model_predict),(20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_line, 2)
            cv2.circle(frame_pro, (12, 12), 5, color_line, -1)
            cv2.imshow("{}".format(video),frame_pro)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_num+=1
            rval,frame_pro=video_data.read()
        video_data.release()
        cv2.destroyAllWindows()
        print("######process {} done, total frame:{}".format(video,frame_num))

    
test()
#python test.py --test_video_path='./UCSD/UCSDped1/Test' --epoch=1 --loss=0.076799
