from keras import backend as K
import tensorflow as tf

def bag_accuracy(y_true,y_pred):
    y_true=K.mean(y_true,axis=0,keepdims=False)
    y_pred=K.mean(y_pred,axis=0,keepdims=False)
    acc=K.mean(K.equal(y_true,K.round(y_pred)))
    return acc
    
def bag_loss(y_true,y_pred):
    y_true_m=K.mean(y_true,axis=0,keepdims=False)
    y_pred_m=K.mean(y_pred,axis=0,keepdims=False)
    '''
    y_true=y_true
    y_pred=y_pred
    print("y_true:",y_true)
    print("y_pred:",y_pred)
    a_min=0
    n_max=0
    a_ind=(K.argmax(y_true,axis=1))
    for ii in (a_ind):
        if(y_pred[int(ii)][0]<a_min):
            a_min=y_pred[int(ii)][0]
    n_ind=(K.argmin(y_true,axis=1))
    for ii in n_ind:
        if(y_true[int(ii)][0]>n_max):
            n_max=y_true[int(ii)][0]
    '''
    loss=K.mean(K.binary_crossentropy(y_true_m,y_pred_m),axis=-1)+K.mean(K.maximum(1. - y_true * y_pred+(1-y_true)*y_pred, 0.), axis=-1)
    return loss
