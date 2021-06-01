import torch
import torch.nn as nn
import torch.nn.functional as F

class Memory(nn.Module):
    def __init__(self,memory_size,feature_dim,memory_dim,tmp_update=0.1,tmp_gather=0.1):
        super(Memory,self).__init__()
        self.memory_size=memory_size
        self.memory_dim=memory_dim
        self.feature_dim=feature_dim
        self.tmp_update=tmp_update
        self.tmp_gather=tmp_gather
        
    def forward(self,query,keys,train=True):
        #print("################model/mem_layer.py, line 15, query.size:{}".format(query.size()))
        batch_size,cpd_channels,t_length,dims=query.size()
        query=F.normalize(query,dim=1)
        
        #train
        if train:
            #loss
            separateness_loss,compactness_loss=self.gather_loss(query,keys,train)
            #read
            updated_query,score_query,score_memory=self.read(query,keys)
            #update
            updated_memory=self.update(query,keys,train)
            #return
            return updated_query,updated_memory,score_query,score_memory,separateness_loss,compactness_loss
        #test
        else:
            #loss
            compactness_loss,query_reshape,top1_key,key_indice=self.gather_loss(query,keys,train)
            #read
            updated_query,score_query,score_memory=self.read(query,keys)
            #update
            updated_memory=keys
            #return
            return updated_query,updated_memory,score_query,score_memory,query_reshape,top1_key,key_indice,compactness_loss
            
    def gather_loss(self,query,keys,train):
        batch_size,cpd_channels,t_length,dims=query.size()
        loss_mse=torch.nn.MSELoss()
        score_query,score_memory=self.get_score(keys,query)
        query_reshape=query.contiguous().view(batch_size*cpd_channels*t_length,dims)
        if train:
            loss=torch.nn.TripletMarginLoss(margin=1.0)
            _,gathering_indices=torch.topk(score_memory,2,dim=1)
            pos=keys[gathering_indices[:,0]]
            neg=keys[gathering_indices[:,1]]
            top1_loss=loss_mse(query_reshape,pos.detach())
            gathering_loss=loss(query_reshape,pos.detach(),neg.detach())
            return gathering_loss,top1_loss
        else:
            _,gathering_indices=torch.topk(score_memory,1,dim=1)
            gathering_loss=loss_mse(query_reshape,keys[gathering_indices].squeeze(1).detach())
            return gathering_loss,query_reshape,keys[gathering_indices].squeeze(1).detach(),gathering_indices[:,0]
            
    def read(self,query,keys):
        batch_size,cpd_channels,t_length,dims=query.size()
        score_query,score_memory=self.get_score(keys,query)
        query_reshape=query.contiguous().view(batch_size*cpd_channels,t_length,dims)
        concat_memory=torch.matmul(score_memory.detach(),keys)
        #updated_query=torch.cat((query_reshape,concat_memory),dim=1)
        #updated_query=updated_query.view(batch_size,t_length,2*dims)
        updated_query=concat_memory.view(batch_size,cpd_channels,t_length,dims)
        return updated_query,score_query,score_memory
        
    def update(self,query,keys,train):
        batch_size,cpd_channels,t_length,dims=query.size()
        score_query,score_memory=self.get_score(keys,query)
        query_reshape=query.contiguous().view(batch_size*cpd_channels*t_length,dims)
        _,gathering_indices=torch.topk(score_memory,1,dim=1)
        _,updating_indices=torch.topk(score_query,1,dim=0)
        query_update=self.get_update_query(keys,gathering_indices,updating_indices,score_query,query_reshape,train)
        updated_memory=F.normalize(query_update+keys,dim=1)
        return updated_memory.detach()
    
    def get_score(self,mem,query):
        bs,cpd_channels,t_length,dims=query.size()
        m,d=mem.size()
        score=torch.matmul(query,torch.t(mem))
        score=score.view(bs*cpd_channels*t_length,m)
        score_query=F.softmax(score,dim=0)
        score_memory=F.softmax(score,dim=1)
        return score_query,score_memory
    
    def get_update_query(self,mem,max_indices,update_indices,score,query,train):
        m,d=mem.size()
        query_update=torch.zeros((m,d)).cuda()
        for i in range(m):
            idx=torch.nonzero(max_indices.squeeze(1)==i)
            a,_=idx.size()
            if a!=0:
                query_update[i]=torch.sum(((score[idx,i]/torch.max(score[:,i]))*query[idx].squeeze(1)),dim=0)
            else:
                query_update[i]=0
        return query_update
