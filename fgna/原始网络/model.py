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

with open('config.yml','r') as yamlfile:
    cfg=yaml.load(yamlfile)
 
img_W=cfg['img_W']
img_H=cfg['img_H']    
time_length=cfg['time_length']
loss=cfg['loss']#mse,bag_loss
optimizer=cfg['optimizer']#sgd,adam
weight_decay=cfg['init_lr']
useGated=cfg['useGated']

def load_model():
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
    
def compile_model(model):
    model.summary()#model parameter of each layer
    if optimizer=='sgd':
        opt=optimizers.SGD(nesterov=True)
    else:
        opt=optimizers.Adam(lr=float(weight_decay), beta_1=0.9, beta_2=0.999)
    model.compile(loss=bag_loss,optimizer=opt,metrics=[bag_accuracy])
