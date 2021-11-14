
import pandas as pd
import glob
import os
import numpy as np
import random
from tqdm import tqdm
from data_load.dataloader_liu import *
from models.model_liu import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision
from torch.autograd import Variable

from utils import read_meta, parse_label, read_across_movie_meta
from utils import get_topk, get_mAP, affmat2retdict, affmat2retlist
from utils import read_affmat_of_one_movie, read_affmat_across_movies
from utils import lp, ccpp
from utils import gpu_lp, gpu_ccpp

def run_ccpp(ct_affmat, tt_affmat, gpu_id):
    """
    ct_affmat: face affinity score (num_person, num_tracklet)
    tt_affmatL body affinity score (num_tracklet, num_tracklet)
    gpu_id: using gpu 0 or cpu -1
    """
    n_cast, n_instance = ct_affmat.shape # (num_person, num_tracklet)
    n_sample = n_cast + n_instance # num_node
    W = np.zeros((n_sample, n_sample))
    W[:n_cast, n_cast:] = ct_affmat
    W[n_cast:, :n_cast] = ct_affmat.T
    W[n_cast:, n_cast:] = tt_affmat
    Y0 = np.zeros((n_sample, n_cast))
    for i in range(n_cast):
        Y0[i, i] = 1
    if gpu_id < 0:
        result = ccpp(W, Y0)
    else:
        result = gpu_ccpp(W, Y0, gpu_id=gpu_id)
    
    return result



path = '../'
out_file = "./output"
model_save_path = "checkpoint/body"
transform = transforms.Compose([
                transforms.ToPILImage(),
                # transforms.Pad((0,40), fill=0, padding_mode='constant'),
                transforms.Resize((227,227)),
                # transforms.Resize((128,256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

test_data = dl_test_dataset(path, train=False,transform=transform)

model = r_net50()
cuda = True
if cuda:
    model.cuda()

model.load_state_dict(torch.load(os.path.join(
                        model_save_path, '6_0.20049910317398426_model.pth')))

model.eval()
test_data_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=test_data.collate_fn, num_workers=2)

trange = tqdm(enumerate(test_data_dataloader),
                    total=len(test_data_dataloader),
                    desc="Test")

cos_similarity = nn.CosineSimilarity(dim=1).cuda()


result_id = {}
result_value = {}
### Start Testing ###
for i, batch in trange:
    if i >= len(test_data_dataloader):
    # if i >= 2:
        break

    movie_cast_img, movie_cast_id, movie_candidate_img, movie_candidate_id, movie_name = batch
    one_result_id = {}
    one_result_value = {}

    for cast_id in movie_cast_id:
        one_result_id[cast_id] = []
        one_result_value[cast_id] = []
    
    if cuda:
        cast_img = Variable(movie_cast_img).cuda()
        candidate_img = Variable(movie_candidate_img)
    
    with torch.no_grad():
        ### Extract cast image feature ###
        _, cast_feature = model(cast_img)
        cast_feature_np = cast_feature.cpu().numpy()
        np.save("feature_save/cast/"+movie_name, cast_feature_np)

        # cast_feature_np = np.load("feature_save/cast/"+movie_name+".npy")
        # cast_feature = torch.from_numpy(cast_feature_np)

        ### Extract candidate image feature ###
        candidate_feature = []
        for idx in range(candidate_img.size(0)):
            _, one_candidate_feature = model(candidate_img[idx].cuda().unsqueeze(0))
            candidate_feature.append(one_candidate_feature)
        candidate_feature = torch.cat(candidate_feature, dim=0)
        candidate_feature_np = candidate_feature.cpu().numpy()
        np.save("feature_save/candidate/"+movie_name, candidate_feature_np)

        # candidate_feature_np = np.load("feature_save/candidate/"+movie_name+".npy")
        # candidate_feature = torch.from_numpy(candidate_feature_np)
        
        #     for index in range(len(similarity)):
        #         if similarity[index] > 0.1:
        #             one_result_id[movie_cast_id[index]].append(movie_candidate_id[idx])
        #             one_result_value[movie_cast_id[index]].append(similarity[index])


        ### Calculate cosine similarity between each cast feature and all candidate img feature
        ct_affmat = []
        for cast_idx in range(cast_feature.size(0)):
            cast_rep_feature = cast_feature[cast_idx].unsqueeze(0).repeat(candidate_feature.size(0),1)
            similarity = cos_similarity(cast_rep_feature, candidate_feature).unsqueeze(0)
            ct_affmat.append(similarity)

        tt_affmat = []
        ### Calculate cosine similarity between each candidate feature and all candidate img feature
        for cand_idx in range(candidate_feature.size(0)):
            cand_feature = candidate_feature[cand_idx].unsqueeze(0).repeat(candidate_feature.size(0),1)
            similarity = cos_similarity(cand_feature, candidate_feature).unsqueeze(0)
            tt_affmat.append(similarity)
            
        ct_affmat = torch.cat(ct_affmat, dim=0)
        tt_affmat = torch.cat(tt_affmat, dim=0)
        
        ### Run PPCC algorithm ###
        tnum = candidate_feature.size(0)
        pids = movie_cast_id
        result = run_ccpp(ct_affmat.cpu().numpy(), tt_affmat.cpu().numpy(), 0)
        ret_dict = affmat2retdict(result, pids)

        ### Transfer candidate id index(int) to candidate id(str)
        one_result_id = {}
        for k, _ in ret_dict.items():
            cast_candidate_index = ret_dict[k]
            candidate_index_to_id = []
            for cand_idx in cast_candidate_index:
                candidate_index_to_id.append(movie_candidate_id[cand_idx])            
            one_result_id[k] = candidate_index_to_id

        result_id.update(one_result_id)


import json    
with open("output/test.json", 'w') as f:
    json.dump(result_id, f)
      
import csv
submission = open(os.path.join(out_file, "submit_test_propagation.csv"), "w+") # "./{}.csv".format(target_dataset_name)
s = csv.writer(submission,delimiter=',',lineterminator='\n')
s.writerow(["Id","Rank"])

for mid, candidate in result_id.items():
    # print(candidate)
    s.writerow([mid, ' '.join(candidate)])

submission.close()




    

