import pandas as pd
from tqdm import tqdm
import numpy as np
import cv2
import math 
import glob
import os
import matplotlib.pyplot as plt
import time
import random
import sys
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision
from torch.autograd import Variable


from data_load.dataloader_liu import *
from models.model_liu import *


path = "../"
cuda = True
lr = 1e-4
batch_size = 64
n_epoch = 1000

manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed(manual_seed)

transform = transforms.Compose([
                transforms.ToPILImage(),
                # transforms.Pad((0,40), fill=0, padding_mode='constant'),
                transforms.Resize((227,227)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

train_data = dl_bing(path, train=True,transform=transform)
valid_data = dl_val_dataset(path, train=False,transform=transform)

model = r_net50()
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_class = torch.nn.NLLLoss()

if cuda:
    model.cuda()
    loss_class.cuda()

log = {}
log["valid_acc"] = []

best_acc = 0
best_epoch = 0
batch_size = 64

def train():
    torch.cuda.empty_cache()
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)#, collate_fn=train_data.collate_fn

    model.train()
    trange = tqdm(enumerate(train_dataloader),
                      total=len(train_dataloader),
                      desc="Train")
    train_loss = 0
    n_correct, n_total = 0,0
    for i, batch in trange:
        if i >= len(train_dataloader):
        # if i >= 2:
            break
        
        optimizer.zero_grad()

        imgs, labels = batch
        if cuda:
            imgs = Variable(imgs).cuda()
            labels = Variable(labels).cuda()

        
        class_out, img_feature = model(imgs)

        err_s_label = loss_class(class_out, labels)
        train_loss += err_s_label.item()

        err_s_label.backward()
        #err_s_label.backward()
        optimizer.step()

        pred = class_out.data.max(1, keepdim=True)[1]
        pred = pred.view((-1, len(imgs)))
        
        n_correct += pred.eq(labels.data.view_as(pred)).cpu().sum().item()
        n_total += len(imgs)
        ACC = n_correct/n_total

        iter_ = 'Loss: %.4f ACC: %.4f'% \
                (train_loss/(i+1), ACC)
        trange.set_postfix_str(s=iter_, refresh=True)

def eval():
    torch.cuda.empty_cache()
    model.eval()
    valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=1, shuffle=False, collate_fn=valid_data.collate_fn, num_workers=8)

    trange = tqdm(enumerate(valid_dataloader),
                      total=len(valid_dataloader),
                      desc="Valid")
    
    cos_similarity = nn.CosineSimilarity(dim=1).cuda()

    valid_loss = 0
    n_correct, n_total = 0,0
    for i, batch in trange:
        if i >= len(valid_dataloader):
        # if i >= 2:
            break

        ### Get the validation data including cast and candidate in a movie
        movie_cast_img, movie_cast_label, movie_candidate_img, movie_candidate_label = batch
        predict_id = []
        predict_value = []

        if cuda:
            cast_img = Variable(movie_cast_img).cuda()
            candidate_img = Variable(movie_candidate_img).cuda()
        
        
        with torch.no_grad():
            ### Extract cast image feature ###
            _, cast_feature = model(cast_img)


            ### Extract candidate image feature ###
            candidate_feature = []
            for idx in range(candidate_img.size(0)):
                _, one_candidate_feature = model(candidate_img[idx].unsqueeze(0))
                candidate_feature.append(one_candidate_feature)
            candidate_feature = torch.cat(candidate_feature, dim=0)

            ### Calculate cosine similarity between each candidate feature and all cast img feature
            for idx in range(candidate_feature.size(0)):
                cand_feature = candidate_feature[idx].unsqueeze(0).repeat(cast_feature.size(0),1)
                similarity = cos_similarity(cast_feature, cand_feature)

                # print(len(movie_cast_label), similarity.size())
                # exit()

                max_value, max_index = torch.max(similarity, 0) ### get the max value from the similarity score
                max_value = max_value.cpu().data.item()
                max_index = max_index.cpu().data.item()
                
                if max_value > 0.1: ### a threshold (need to change)
                    predict_id.append(movie_cast_label[max_index])
                else:
                    predict_id.append("others")
                predict_value.append(max_value)
     
        n_total+=len(movie_candidate_label)
        for idx in range(len(movie_candidate_label)):
            if movie_candidate_label[idx] == predict_id[idx]:
                n_correct+=1

        ACC = n_correct/n_total

        iter_ = 'ACC: %.4f'% (ACC)

        
        trange.set_postfix_str(s=iter_, refresh=True)
    return ACC


def main():
    best_acc = 0
    best_epoch = 0
    for epoch in range(n_epoch):
        ### train the model ###
        train()
        ### eval the model ###
        ACC = eval()
        log["valid_acc"].append(ACC)
        print('epoch: %d, accuracy: %f' % (epoch, ACC))

        if ACC > best_acc:
            print("New record Achieve")
            best_acc = ACC
            best_epoch = epoch
            torch.save(model.state_dict(), 'checkpoint/body/{1}_{2}_model.pth'.format("checkpoint_1", epoch, best_acc))

        print(best_epoch, ": ", best_acc)
        



if __name__ == "__main__":
    main()


            
