import yaml
import numpy as np
from keras.callbacks import ModelCheckpoint,EarlyStopping
import matplotlib.pyplot as plt
from keras.utils.io_utils import HDF5Matrix
from model import load_model,compile_model
import os

with open('config.yml','r') as ymlfile:
    cfg=yaml.load(ymlfile)

epochs=cfg['epochs']
batch_size=cfg['batch_size']

def train():
    print("----------build model----------")
    model=load_model()
    print('----------complie model----------')
    compile_model(model)

    train_folder='train'
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    print('----------load data----------')
    train_data=HDF5Matrix('train.h5','data')
    train_gt=HDF5Matrix('gt_train.h5','data')
    print("#######train_data.shape:{}".format(train_data.shape))
    print("#######train_gt.shape:{}".format(train_gt.shape))
        
    snapshot=ModelCheckpoint(os.path.join(train_folder,'model_snapshot_epoch{epoch:04d}_loss{loss:.6f}.h5'))
    earlystop=EarlyStopping(monitor='loss',patience=5)
    print('----------start train----------')
    history=model.fit(
        train_data,
        train_gt,
        batch_size=batch_size,
        epochs=epochs,
        shuffle='batch',
        callbacks=[snapshot,earlystop])
    print('----------train completed----------')
    np.save(os.path.join(train_folder,'train_profile.npy'),history.history)
    epoch_num=len(history.history['loss'])
    print('----------plot epoch loss----------')
    plt.plot(range(1,epoch_num+1),history.history['loss'],'g--',label='train_loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('train_loss.png')

def retrain():
    import argparse
    from keras.models import load_model
    from sp_layer import Mil_Attention, Bag_Pooling
    from keras.regularizers import l2
    from sp_loss import bag_accuracy, bag_loss

    parser=argparse.ArgumentParser(description='Base Path')
    parser.add_argument('--epoch',type=str)
    parser.add_argument('--loss',type=str)
    args=parser.parse_args()

    name_epoch=args.epoch
    name_loss=args.loss
    
    print("----------build model----------")
    model_name="./train/model_snapshot_epoch{:04d}_loss{}.h5".format(int(name_epoch),name_loss)
    model=load_model(model_name,custom_objects={'Mil_Attention' : Mil_Attention, 'Bag_Pooling': Bag_Pooling,'bag_loss' : bag_loss, 'bag_accuracy':bag_accuracy})
    train_folder='train'
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    print('----------load data----------')
    train_data=HDF5Matrix('train.h5','data')
    train_gt=HDF5Matrix('gt_train.h5','data')
    print("#######train_data.shape:{}".format(train_data.shape))
    print("#######train_gt.shape:{}".format(train_gt.shape))
        
    snapshot=ModelCheckpoint(os.path.join(train_folder,'model_snapshot_epoch{epoch:04d}_loss{loss:.6f}.h5'))
    earlystop=EarlyStopping(monitor='loss',patience=5)
    print('----------start train----------')
    history=model.fit(
        train_data,
        train_gt,
        batch_size=batch_size,
        epochs=epochs,
        shuffle='batch',
        initial_epoch=30,
        callbacks=[snapshot,earlystop])
    print('----------train completed----------')
    np.save(os.path.join(train_folder,'train_profile.npy'),history.history)
    epoch_num=len(history.history['loss'])
    print('----------plot epoch loss----------')
    plt.plot(range(1,epoch_num+1),history.history['loss'],'g--',label='train_loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('train_loss.png')


train()
#python train.py

#retrain()
#python train.py --epoch=1 --loss=0.618812
