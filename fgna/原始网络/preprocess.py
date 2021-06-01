import argparse
import os
import numpy as np
import h5py
from tqdm import tqdm  
import yaml

parser=argparse.ArgumentParser(description='Base Path')
parser.add_argument('--base_path',type=str)
parser.add_argument('--style',type=str)
args=parser.parse_args()

base_path=args.base_path #/avenue-train_video,test_video
style=args.style #train or test

with open('config.yml','r') as yamlfile:
    cfg=yaml.load(yamlfile)

img_W=cfg['img_W']
img_H=cfg['img_H']
time_length=cfg['time_length']
img_size=(img_W,img_H)

def video_to_frame():
    import skvideo.io
    from skimage.transform import resize
    from skimage.io import imsave
    #convert video to frames in train or test filefolder 
    #video_to_frame()
    #python preprocess.py --base_path='/home/test/smyWork/test' --style=train
    video_path=os.path.join(base_path,'{}_video'.format(style));
    frame_path=os.path.join(base_path,'{}_frame'.format(style));
    if not os.path.exists(frame_path):
        os.makedirs(frame_path)
    video_count=0
    for video_file in os.listdir(video_path):        
        if video_file.lower().endswith(('.avi','.mp4')):
            video_count+=1
            video_file_path=os.path.join(video_path,video_file)#/base_path/test_video/1.avi
            video_frame_path=os.path.join(frame_path,video_file.split('.')[0])#/base_path/{}_frame/1/
            if not os.path.exists(video_frame_path):
                os.makedirs(video_frame_path)
            print('###### convert '+video_file_path+' to pics')
            videocap=skvideo.io.vreader(video_file_path)
            frame_count=1
            for image in videocap:
                image=resize(image,img_size,mode='reflect')
                imsave(os.path.join(video_frame_path,'{}_{:05d}.jpg'.format(video_file.split('.')[0],frame_count)),image)
                frame_count+=1
    print("###### video_to_frame() done, process {} videos".format(video_count))
    return frame_path
                
def video_process_data(video_frame_path):
    from keras.preprocessing.image import load_img,img_to_array
    from skimage.transform import resize
    #process single video images and store data in npy
    #video_process_data('/home/test/smyWork/avenue/testing_frames/01')
    #python preprocess.py --base_path='/home/test/smyWork/avenue' --style=train
    print("###### load "+video_frame_path+" to process")
    imagestore=[]
    img_files=os.listdir(video_frame_path)
    img_files.sort()
    for img_file in img_files:#/base_path/{}_frame/1/
        if img_file.endswith(('.jpg','.png','.tif')):
            img_file_path=os.path.join(video_frame_path,img_file);
            #print("###### load "+img_file_path+" to process")
            img=load_img(img_file_path)
            img=img_to_array(img)
            img=resize(img,img_size)
            gray=0.2989*img[:,:,0]+0.5870*img[:,:,1]+0.1140*img[:,:,2]
            imagestore.append(gray)
    imagestore=np.array(imagestore)#num,i_w,i_h
    #num,i_w,i_h=imagestore.shape
    #imagestore.resize(i_w,i_h,num)
    imagestore=(imagestore-imagestore.mean()+imagestore.std())/(2*imagestore.std())
    imagestore=np.clip(imagestore,0,1)#make negative values to 0, make the maximum to 1
    video_name=video_frame_path.split('/')[-1]
    #print("###### imagestore.shape=",imagestore.shape)
    #print("###### video_name=",video_name)
    video_npys_save_path=os.path.join('/'.join(video_frame_path.split('/')[:-2]),'{}_npys'.format(style))
    if not os.path.exists(video_npys_save_path):
        os.makedirs(video_npys_save_path)
    video_npy_save_path=os.path.join(video_npys_save_path,'video_{}_{}.npy'.format(video_name,style))
    np.save(video_npy_save_path,imagestore)
    print("video_proces_data() done, save {}".format(video_npy_save_path))#/base_path/train_npys/video_01_train.npy
    return video_npys_save_path
    
def video_build_h5(video_npys_save_path):
    #time_length frames consist a batch
    #video_build_h5('/home/test/smyWork/avenue/train_npys')
    ##python preprocess.py --base_path='/home/test/smyWork/avenue' --style=train
    #import h5py
    #from tqdm import tqdm    
    videos_npys_data=os.listdir(video_npys_save_path)#/base_path/train_npys
    videos_npys_data.sort()
    for video_npy_data_path in tqdm(videos_npys_data):
        video_npy_data=np.load(os.path.join(video_npys_save_path,video_npy_data_path))
        print("###### translate {} to h5".format(video_npy_data_path))
        video_npy_data=np.expand_dims(video_npy_data,axis=-1)
        video_frame_num=video_npy_data.shape[0]
        #print("###### video_frame_num={}".format(video_frame_num))
        video_patch_data=np.zeros((video_frame_num-time_length,time_length,img_W,img_H,1)).astype('float16')
        for vol_num in range(video_frame_num-time_length):
            video_patch_data[vol_num]=video_npy_data[vol_num:vol_num+time_length]
        h5_store_path=os.path.join('/'.join(video_npys_save_path.split('/')[:-1]),'{}_h5s'.format(style))
        if not os.path.exists(h5_store_path):
            os.makedirs(h5_store_path)
        with h5py.File(os.path.join(h5_store_path,'{}_h5.h5'.format(video_npy_data_path.split('.')[0])),'w') as f:
            #h5_store_path=/base_path/train_h5s
            #np.random.shuffle(video_patch_data)
            f['data']=video_patch_data
    print("###### video_build_h5() done.")
    return h5_store_path
    
def create_dataset(h5_store_path):
    #video_build_h5('/home/test/smyWork/avenue/train_h5s')
    ##python preprocess.py --base_path='/home/test/smyWork/avenue' --style=train
    #import h5py
    #from tqdm import tqdm
    #dataset_path=os.path.join('/'.join(h5_store_path.split('/')[:-1]))
    dataset_path=os.path.dirname(os.path.realpath(__file__))
    #if not os.path.exists(dataset_path):
    #    os.makedirs(dataset_path)
    dataset_file=h5py.File('{}.h5'.format(style),'w')
    #dataset_file=h5py.File(os.path.join(dataset_path,'{}.h5'.format(style)),'w')
    h5s_filelist=sorted([os.path.join(h5_store_path,item) for item in os.listdir(h5_store_path)])
    total_rows=0
    for h5_file_num,h5_file in enumerate(tqdm(h5s_filelist)):
        h5_data_file=h5py.File(h5_file,'r')
        h5_data=h5_data_file['data']
        total_rows+=h5_data.shape[0]
        print("######h5-data.shape={}".format(h5_data.shape[0]))
        print("######total_rows={}".format(total_rows))
        if h5_file_num==0:
            create_dataset=dataset_file.create_dataset('data',(total_rows,time_length,img_W,img_H,1),maxshape=(None,time_length,img_W,img_H,1))
            create_dataset[:,:]=h5_data
            append_position=total_rows
        else:
            create_dataset.resize(total_rows,axis=0)
            create_dataset[append_position:total_rows,:]=h5_data
            append_position=total_rows
    dataset_file.close()
    print("###### create_dataset() done, store {}".format(dataset_path))
    return dataset_path

def gt_build_h5(gt_file_path):
    #time_length frames consist a batch
    #gt_build_h5('./gt')
    ##python preprocess.py --base_path='./ucsdped1' --style=train
    #import h5py
    #from tqdm import tqdm    
    gt_data=os.listdir(gt_file_path)#/base_path/train_npys
    gt_data.sort()
    for gt_data_path in tqdm(gt_data):
        gt_file_data=[]
        with open(os.path.join(gt_file_path,gt_data_path)) as f:
            lines=f.readlines()
            for line in lines:
                line=line.rstrip()
                line=line.split(' ')[1]
                gt_file_data.append(line)
        gt_file_data=np.array(gt_file_data,dtype='int')
        np.transpose(gt_file_data)
        print("gt_file_data.shape:{}".format(gt_file_data.shape))
        print("###### translate {} to h5".format(gt_data_path))
        gt_file_data=np.expand_dims(gt_file_data,axis=-1)
        video_frame_num=gt_file_data.shape[0]
        #print("###### video_frame_num={}".format(video_frame_num))
        gt_patch_data=np.zeros((video_frame_num-time_length,time_length,1)).astype('float16')
        for vol_num in range(video_frame_num-time_length):
            gt_patch_data[vol_num]=gt_file_data[vol_num:vol_num+time_length]
        h5_store_path=os.path.join('/'.join(gt_file_path.split('/')[:-1]),'gt_{}_h5s'.format(style))
        if not os.path.exists(h5_store_path):
            os.makedirs(h5_store_path)
        with h5py.File(os.path.join(h5_store_path,'{}_h5.h5'.format(gt_data_path.split('.')[0])),'w') as f:
            #h5_store_path=/base_path/train_h5s
            #np.random.shuffle(video_patch_data)
            f['data']=gt_patch_data
    print("###### video_build_h5() done. save {}".format(h5_store_path))
    print("###### gt_patch_data.shape:{}".format(gt_patch_data.shape))
    return h5_store_path
    
def gt_create_dataset(gt_file_path):
    #process gt file into h5 dataset
    #gt_create_dataset('./gt')
    #python preprocess.py --base_path='./ucsdped1' --style=train
    gt_dataset_path=os.path.dirname(os.path.realpath(__file__))
    gt_dataset_file=h5py.File('gt_train_timelen.h5')
    gt_filelist=sorted([os.path.join(gt_file_path,item) for item in os.listdir(gt_file_path)])
    total_rows=0
    
    for gt_file_num,gt_file in enumerate(tqdm(gt_filelist)):
        print("###### load "+gt_file+" to process")
        h5_data_file=h5py.File(gt_file,'r')
        h5_data=h5_data_file['data']
        total_rows+=h5_data.shape[0]
        print("######h5-data.shape={}".format(h5_data.shape[0]))
        print("######total_rows={}".format(total_rows))
        if gt_file_num==0:
            create_dataset=gt_dataset_file.create_dataset('data',(total_rows,time_length,1),maxshape=(None,time_length,1))
            create_dataset[:,:]=h5_data
            append_position=total_rows
        else:
            create_dataset.resize(total_rows,axis=0)
            create_dataset[append_position:total_rows,:]=h5_data
            append_position=total_rows
        print("###### create_dataset.shape:{}".format(create_dataset.shape))
        print("append_position:{}".format(append_position))
    gt_dataset_file.close()
    print("###### gt_create_dataset() done, store {}".format(gt_dataset_path))
    return gt_dataset_path

def gt_create_dataset_alone(gt_file_path):
    #process gt file into h5 dataset
    #gt_create_dataset('./gt')
    #python preprocess.py --base_path='./ucsdped1' --style=train
    gt_dataset_path=os.path.dirname(os.path.realpath(__file__))
    gt_dataset_file=h5py.File('gt_train.h5')
    gt_filelist=sorted([os.path.join(gt_file_path,item) for item in os.listdir(gt_file_path)])
    total_rows=0
    
    for gt_file_num,gt_file in enumerate(tqdm(gt_filelist)):
        print("###### load "+gt_file+" to process")
        gt_file_data=[]
        with open(gt_file) as f:
            lines=f.readlines()
            for line in lines:
                line=line.rstrip()
                line=line.split(' ')[1]
                gt_file_data.append(line)
        h5_data=np.array(gt_file_data,dtype='int')
        h5_data=h5_data[:,np.newaxis]
        total_rows+=h5_data.shape[0]-time_length
        print("######h5-data.shape={}".format(h5_data.shape))
        print("######total_rows={}".format(total_rows))
        if gt_file_num==0:
            create_dataset=gt_dataset_file.create_dataset('data',(total_rows,1),maxshape=(None,1))
            create_dataset[:]=h5_data[:int(h5_data.shape[0]-time_length)]
            append_position=total_rows
        else:
            create_dataset.resize(total_rows,axis=0)
            create_dataset[append_position:total_rows]=h5_data[:int(h5_data.shape[0]-time_length)]
            append_position=total_rows
        print("###### create_dataset.shape:{}".format(create_dataset.shape))
        print("append_position:{}".format(append_position))
    gt_dataset_file.close()
    print("###### gt_create_dataset() done, store {}".format(gt_dataset_path))
    return gt_dataset_path
        
def preprocess_data():
    '''
    #######video data processing#######
    print("----------video to frame----------")
    frame_path=video_to_frame()
    print('==========frame_path:{}=========='.format(frame_path))
    print("----------frame to npy----------")
    for video_frame_path in tqdm(sorted([os.path.join(frame_path,item) for item in os.listdir(frame_path)])):
        video_npys_save_path=video_process_data(video_frame_path)
    print('==========video_npys_save_path:{}=========='.format(video_npys_save_path))
    print("----------npy to h5----------")
    #video_npys_save_path='/home/test/smyWork/test/train_npys'
    h5_store_path=video_build_h5(video_npys_save_path)
    print("==========h5_store_path:{}==========".format(h5_store_path))
    print('---------h5 to dataset----------')
    h5_store_path='/home/test/smyWork/test/train_h5s'
    dataset_file=create_dataset(h5_store_path)
    print("==========dataset_path:{}==========".format(dataset_file))
    print("##########preprocess data done!##########")
    '''
    
    #######file data process#######
    print("----------frame to npy----------")
    for video_frame_path in tqdm(sorted([os.path.join(os.path.join(base_path,'trainset'),item) for item in os.listdir(os.path.join(base_path,'trainset'))])):
        video_npys_save_path=video_process_data(video_frame_path)
    print('==========video_npys_save_path:{}=========='.format(video_npys_save_path))
    print("----------npy to h5----------")
    h5_store_path=video_build_h5(video_npys_save_path)
    print("==========h5_store_path:{}==========".format(h5_store_path))
    print('---------h5 to dataset----------')
    dataset_file=create_dataset(h5_store_path)
    print("==========dataset_path:{}==========".format(dataset_file))
    print('---------gt to dataset----------')
    gt_dataset_path=gt_create_dataset_alone(os.path.join(base_path,'gt'))
    print("==========gt_dataset_path:{}==========".format(gt_dataset_path))
    print("##########preprocess data done!##########")
    
preprocess_data()

#gt_h5_path=gt_build_h5('./ucsdped1/gt')
#gt_create_dataset(gt_h5_path)
#gt_create_dataset('./ucsdped1/gt_train_h5s')

#gt_create_dataset_alone('./gt')

#video_to_frame()

#usage:
#source activate smy-ab2
#python preprocess.py --base_path='/home/test/smyWork/test' --style=train  ###video data process
#python preprocess.py --base_path='./ucsdped1' --style=train   ###file data process
