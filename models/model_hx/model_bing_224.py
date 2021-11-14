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
    def __init__(self):
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
        self.max_c = 11
        self.pred = pred_bing(self.max_c)

    def forward(self, c, x ,c_num, device):
        bs = x.shape[0]

        c = self.query(c)
        c = c.view(-1,2048)
        
        k = self.key(x)
        k = k.view(bs,2048,-1)
        
        v = self.value(x)
        v = v.view(bs,256,-1)
        v_t = v.transpose(1,2).contiguous()
        
        c_num_ac= list(accumulate(c_num))
        q_pad = torch.zeros([bs,self.max_c,2048]).to(device)
        for i in range(bs):
            if i == 0:
                q_pad[i,0:c_num[i]] = c[i:c_num_ac[i]]
            else:
                q_pad[i,0:c_num[i]] = c[c_num_ac[i-1]:c_num_ac[i]]
        
        qk = torch.bmm(q_pad,k)
        qkv = torch.bmm(qk, v_t)
        qkv = qkv.view(bs,self.max_c,16,16).contiguous()
       
        pred = self.pred(qkv)
    
        return pred

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
        
def test():
    model = r_net50()
    qkv = QKV()
    cast = torch.rand(20,3,224,224)
    img = torch.rand(3,3,224,224)
    num = [4,10,6]
    device='cpu'
#    P  = model(cast,img,num, device='cpu')
    FEA_x = model(img)
    FEA_c = model(cast)    
    print(FEA_x.shape, FEA_c.shape)
    pred= qkv(FEA_c, FEA_x ,num, device)
    print(pred.shape)

if __name__ == '__main__':
    test()