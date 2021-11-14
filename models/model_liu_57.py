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
    def __init__(self, pretrained=True, num_class=941):
        super(r_net50, self).__init__()
        #load the model
        # self.model = models.resnet50(pretrained=pretrained)   

        resnet50 = models.resnet50(pretrained=pretrained)
        self.model = torch.nn.Sequential(*(list(resnet50.children())[:-1]))    

        self.classifier = Classifier(num_class, in_dim=2048)
        
    def forward(self, x):
        bs = x.shape[0]
        # x = x.view(-1,3,448,448).contiguous()
        x = self.model(x)
        x = x.view(-1, 2048)
    
        out, feature = self.classifier(x)
        
        return out, feature
        
class r_net101(nn.Module):
    def __init__(self, pretrained=True, num_class=942):
        super(r_net101, self).__init__()
        #load the model
        # self.model = models.resnet50(pretrained=pretrained)   

        resnet101 = models.resnet101(pretrained=pretrained)
        
        self.model = torch.nn.Sequential(*(list(resnet101.children())[:-1]))    
        
        self.classifier = Classifier(num_class, in_dim=2048)
        
    def forward(self, x):
        bs = x.shape[0]
        # x = x.view(-1,3,448,448).contiguous()
        x = self.model(x)
        x = x.view(-1, 2048)

        # return _, x
    
        out, feature = self.classifier(x)
        
        return out, feature


class Classifier(nn.Module):
    def __init__(self, num_class=941, in_dim=2048):
        super(Classifier, self).__init__()

        self.feature = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),

            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
        )

        self.out = nn.Linear(256, num_class)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # feature = self.feature(x)
        # out = self.out(feature)
        # out = self.log_softmax(out)

        module_list = list(self.feature.modules())
        # print("0", module_list[0])
        # print("1",module_list[1])
        # print("2", module_list[2])
        # print("3", module_list[3])
        # print("4", module_list[4])
        # print("5", module_list[5])
        # print(x.size())
        for l in module_list[1:2]:
            # print(l)
            feature = l(x)
            x = feature
            # print(feature.size())
        # exit()
        return None, feature

        return out, feature

class r_net50_512(nn.Module):
    def __init__(self, pretrained=True, num_class=941):
        super(r_net50_512, self).__init__()
        #load the model
        # self.model = models.resnet50(pretrained=pretrained)   

        resnet50 = models.resnet50(pretrained=pretrained)
        self.model = torch.nn.Sequential(*(list(resnet50.children())[:-1]))    

        self.feature = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.Dropout(),
            nn.ReLU(True),
            nn.Linear(1024, 512)
        )
        
        # self.classifier = Classifier(num_class, in_dim=2048)
        
    def forward(self, x):
        bs = x.shape[0]
        # x = x.view(-1,3,448,448).contiguous()
        x = self.model(x)
        x = x.view(-1, 2048)
    
        feature = self.feature(x)
        
        return None, feature

def extract_body_feature(imgs, model, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), tta = True):
    
    flipped = torch.flip(imgs, [3]) 

    model.to(device)
    # extract features
    model.eval() # set to evaluation mode
    with torch.no_grad():
        if tta:
            emb_batch = model(imgs.to(device))[1] + model(flipped.to(device))[1]
            features = l2_norm(emb_batch)
        else:
            features = l2_norm(model(ccropped.to(device)).cpu())

    return features

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output
    
def test():
    model = r_net50()
    img = torch.rand(1,3,227,227)
    out, f = model(img)
    print(out.size())
    print(f.size())
    

if __name__ == '__main__':
    test()