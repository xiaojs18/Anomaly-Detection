# -*- coding: UTF-8 -*-
import os
import numpy as np
import sys

encoder_path='./test_encoder'
#bagAtten_path='./test_bagAtten'
bagAtten_path='./test_20'
vae_path='./test_vae'
groundtruth_file_path='./groundtruth'
gt_file_path='./gt'
pic_file_path='./test_pic'


def pain_pic():
    import matplotlib.pyplot as plt
    if not os.path.exists(pic_file_path):
        os.makedirs(pic_file_path)
    for predict_file_name in os.listdir(encoder_path):
        predict_file=os.path.join(encoder_path,predict_file_name)
        print("###### read {}".format(predict_file))
        pic_file_name=os.path.join(pic_file_path,'{}.png'.format(predict_file_name.split('.')[0]))
        encoder_list=cal_score_encoder(predict_file)
        bagAtten_list=cal_score_bagAtten(os.path.join(bagAtten_path,predict_file_name))
        #vae_list=cal_score_encoder(os.path.join(vae_path,predict_file_name))
        plt.plot(range(1,len(encoder_list)+1),encoder_list,'--',linewidth=4,label='Score_ENC')
        plt.plot(range(1,len(encoder_list)+1),bagAtten_list,'-',linewidth=4,label='Score_GRP')
        #plt.plot(range(1,len(encoder_list)+1),vae_list,linewidth=2,label='test_score_vae')
        plt.xlabel('Frames',{'size':28,'family':'Times New Roman'})
        plt.ylabel('Score',{'size':28,'family':'Times New Roman'})
        plt.ylim(0, 1)
        plt.xlim(1, len(encoder_list)+1)
        #plt.legend(loc=0,prop={'size':24,'family':'Times New Roman'},frameon=False)
        
        
        #####paint gt
        gt_file=os.path.join(groundtruth_file_path,'{}_groundtruth.txt'.format(predict_file_name.split('_')[0]))
        start_list=[]
        ens_list=[]
        with open(gt_file) as f:
            line=f.readline()
            line=line.rstrip()
            start_list=line.split(' ')
            line=f.readline()
            end_list=line.split(' ')
        #print(start_list,end_list)
        for i in range(len(start_list)):
            start=int(start_list[i])
            end=int(end_list[i])
            plt.fill_between(range(start,end),0,1,facecolor='orange',alpha='0.4')
        
        ax=plt.gca()
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        ax.spines['bottom'].set_linewidth(2);
        ax.spines['left'].set_linewidth(2);
        ax.spines['right'].set_linewidth(2);
        ax.spines['top'].set_linewidth(2);
        plt.rc('font',family='Times New Roman',size='24')
        #plt.savefig(pic_file_name)
        #plt.imshow(image)
        plt.show()
        plt.clf() #clear figure axes but don't close the window
        print("######save pic_file_name:{}".format(pic_file_name))
    plt.close()
    

def cal_score_encoder(predict_file):
    data_list=[]
    with open(predict_file) as f:
        lines=f.readlines()
        for line in lines:
            line=line.rstrip()
            line=line.split(' ')[2]
            data_list.append(line)
    data_list=np.array(data_list,dtype=float)
    data_list=data_list-min(data_list)
    data_list=(data_list/float(max(data_list)))
    return data_list
    
def cal_score_bagAtten(predict_file):
    data_list=[]
    with open(predict_file) as f:
        lines=f.readlines()
        for line in lines:
            line=line.rstrip()
            line=line.split(' ')[2]
            data_list.append(line)
    data_list=np.array(data_list,dtype=float)
    return data_list
    
pain_pic()
