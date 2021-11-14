import pandas as pd
import numpy as np
from PIL import Image
import cv2
import math 
import glob
import os
import matplotlib.pyplot as plt
from itertools import accumulate

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import dataloader, dataset
import torch.utils.model_zoo as model_zoo
import torchvision
import torchvision.transforms as T
from torchvision import models


# Define the ResNet50-based Model
class r_net50(nn.Module):
    def __init__(self):
        super(r_net50, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.model = nn.Sequential(*(list(resnet50.children())[:-2]))    

    def forward(self, x):        
        x = self.model(x)   
        return x
#(avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
        
class QKV(nn.Module):
    def __init__(self,max_c = 11):
        super(QKV, self).__init__()
        self.query = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                   nn.Tanh()
                                   )
        self.key = nn.Sequential(nn.Conv2d(2048,1024,3),
                                 nn.BatchNorm2d(1024),
                                 nn.LeakyReLU(0.2),
                                 nn.Dropout(0.5),
                                 nn.Conv2d(1024,512,1),
                                 nn.BatchNorm2d(512),
                                 nn.LeakyReLU(0.2),
                                 nn.Dropout(0.5),
                                 nn.Conv2d(512,2048,1),
                                 nn.Tanh()
                                 )
        self.value = nn.Sequential(nn.Conv2d(2048,1024,3),
                                   nn.BatchNorm2d(1024),
                                   nn.LeakyReLU(0.2),
                                   nn.Dropout(0.5),
                                   nn.Conv2d(1024,512,1),
                                   nn.BatchNorm2d(512),
                                   nn.LeakyReLU(0.2),
                                   nn.Dropout(0.5),
                                   nn.Conv2d(512,512,1),
                                   nn.BatchNorm2d(512),
                                   nn.LeakyReLU(0.2),
                                   nn.Dropout(0.5),
                                   nn.Conv2d(512,256,1),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(0.2)
                                   )
        self.max_c = max_c
        self.pred = pred_bing(self.max_c)
        self.down = kqkv2class(self.max_c)

    def forward(self, c, x ,c_num, device, bs, fs):
        c = self.query(c)
        c = c.view(-1,2048)

        c_num_ac= list(accumulate(c_num))
        q_pad = torch.zeros([bs,self.max_c,2048]).to(device)
        for i in range(bs):
            if i == 0:
                q_pad[i,0:c_num[i]] = c[i:c_num_ac[i]]
            else:
                q_pad[i,0:c_num[i]] = c[c_num_ac[i-1]:c_num_ac[i]]
        
        k = self.key(x)
        k = k.view(bs,fs,2048,-1)
        
        v = self.value(x)
        v = v.view(bs,fs,256,-1)

        qx = self.query(x)
        qx = qx.view(bs,fs,2048)
        
        qkv_ = []
#        tqkv_ = []
        xqkv_ = []
        for i in range(fs):
            ki = k[:,i]
            vi = v[:,i]
            v_t = vi.transpose(1,2).contiguous()       
            qk = torch.bmm(q_pad,ki)
            qkv = torch.bmm(qk, v_t)
            qkv = qkv.view(bs,self.max_c,16,16).contiguous()

            qkv_.append(qkv.unsqueeze(0))
#            tqkv_.append(qkv.view(bs,-1).unsqueeze(0))
            
            #qx seris
            xqk = torch.bmm(qx,ki)
            xqkv = torch.bmm(xqk, v_t)
            xqkv = xqkv.view(bs,fs,16,16).contiguous()
            xqkvm = torch.mean(xqkv,1,keepdim=True)
            
            xqkv_.append(xqkvm.unsqueeze(0))       
        
        tkkv_ = []
        for i in range(bs):
            ki = k[i]
            kii = torch.mean(ki,2)
            k_ti = kii.transpose(0,1).contiguous()
            vi = v[i]
            vii = torch.mean(vi,2)

            kki = torch.matmul(kii,k_ti)
            kkvi = torch.matmul(kki,vii)
            kkvi = kkvi.view(fs,1,16,16).contiguous()
            tkkv_.append(kkvi.unsqueeze(0))
        tkkv =  torch.cat(tkkv_,0).contiguous()
                
        pred_ = torch.cat(qkv_,0).transpose(0,1).contiguous()
        x_sim_ = torch.cat(xqkv_,0).transpose(0,1).contiguous()
        c_pred_ = torch.cat([pred_,x_sim_,tkkv],2)
        c_pred_ = c_pred_.view(bs*fs,self.max_c+1+1,16,16)
        
        kqkv = self.down(c_pred_)
        pred = self.pred(kqkv)
#        tqkv = torch.cat(tqkv_,0).contiguous()
        kqkv = kqkv.view(bs,fs,-1).transpose(0,1).contiguous()
    
        return pred, kqkv

class kqkv2class(nn.Module):
    def __init__(self,max_c):
        super(kqkv2class, self).__init__()
        self.downs = nn.Sequential(nn.Conv2d(11+1+1,256,1,padding=0),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(0.2),
                                   nn.Dropout(0.5),
                                   nn.Conv2d(256,max_c,3,padding=1),
                                   nn.LeakyReLU(0.2),
                                   nn.Dropout(0.5),
                                   )
        self.max_c = max_c
        
    def forward(self, x):
        x = self.downs(x)
        return x

class pred_bing(nn.Module):
    def __init__(self,max_c):
        super(pred_bing, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(max_c,1024,3),
                                   nn.BatchNorm2d(1024),
                                   nn.LeakyReLU(0.2),
                                   nn.Dropout(0.5),
                                   nn.Conv2d(1024,512,3,2),
                                   nn.BatchNorm2d(512),
                                   nn.LeakyReLU(0.2),
                                   nn.Dropout2d(0.5),
                                   nn.Conv2d(512,256,3),
                                   nn.BatchNorm2d(256),
                                   nn.Dropout2d(0.5),
                                   nn.Conv2d(256,max_c,4)
                                   )
        self.max_c = max_c
        
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1,self.max_c)
        return x

class temp_bing(nn.Module):
    def __init__(self):
        super(temp_bing, self).__init__()
        self.rnn = nn.GRU(2816,2816,2,batch_first=False,dropout=0.2)
        
        self.att = nn.Sequential(nn.Linear(2816,1024),
                                 nn.BatchNorm1d(1024),
                                 nn.LeakyReLU(0.2),
                                 nn.Dropout(0.5),
                                 nn.Linear(1024,512),
                                 nn.BatchNorm1d(512),
                                 nn.LeakyReLU(0.2),
                                 nn.Dropout(0.5),
                                 nn.Linear(512,2816),
                                 nn.Dropout(0.5),
                                 nn.Sigmoid())

        self.max_c = 11
        self.pred = pred_bing(self.max_c)
        
    def forward(self, x, hn, bs, fs):
        x_att = x.view(-1,2816)
        x_att = self.att(x_att)   
        x_att = x_att.view(fs,bs,2816)
        x, hn = self.rnn(x*x_att)
        x = x.transpose(0,1).contiguous().view(bs*fs,self.max_c,16,16)
        pred = self.pred(x)
        return pred, hn


        
def test():
    model = r_net50()
    qkv = QKV()
    tem = temp_bing()
    
    cast = torch.rand(20,3,224,224)
    img = torch.rand(15,3,224,224)
    bs = 3
    fs = 5
    
    num = [4,10,6]
    device='cpu'

    FEA_x = model(img)
    FEA_c = model(cast)    
    print(FEA_c.shape,FEA_x.shape)
    pred, tqkv= qkv(FEA_c, FEA_x ,num, device, bs,fs)
#    qkv = qkv.view(3,3,-1)
    print(pred.shape, tqkv.shape)
    ttt, hn = tem(tqkv,hn=None,bs=bs,fs=fs)
    print(ttt.shape)

if __name__ == '__main__':
    test()