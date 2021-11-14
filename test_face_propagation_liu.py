
import pandas as pd
import glob
import os
import numpy as np
import random
from tqdm import tqdm
from data_load.dataloader_liu import *
from models.model_liu import *
from models.model_irse import *
import cv2
import pickle

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

from detect import detect




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
out_file = "./output/face_ppcc"
model_body_save_path = "checkpoint/body"
model_face_save_path = "checkpoint/face3"
transform = transforms.Compose([
                transforms.ToPILImage(),
                # transforms.Pad((0,40), fill=0, padding_mode='constant'),
                transforms.Resize((227,227)),
                # transforms.Resize((128,256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
transform_face = transforms.Compose([
                transforms.ToPILImage(),
                # transforms.Pad((0,40), fill=0, padding_mode='constant'),
                transforms.Resize((112,112)),
                # transforms.Resize((128,256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


test_data = dl_test_dataset(path, train=False,transform=transform)

model_body = r_net50()
# model_face = r_net50()
model_face = IR_50([112,112])
cuda = True
if cuda:
    model_body.cuda()
    model_face.cuda()

model_body.load_state_dict(torch.load(os.path.join(
                        model_body_save_path, '6_0.20049910317398426_model.pth')))
# model_face.load_state_dict(torch.load(os.path.join(
#                         model_face_save_path, '2_0.15695755023525437_model.pth')))
model_face.load_state_dict(torch.load("./models/backbone_ir50_ms1m_epoch120.pth"))


model_body.eval()
model_face.eval()
test_data_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=test_data.collate_fn, num_workers=0)

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
    # movie_cast_img, movie_cast_id, movie_candidate_img, movie_candidate_id, movie_name = batch
    movie_cast_img, movie_cast_id, movie_candidate_img, movie_candidate_id, movie_name, movie_cast_detect_img, movie_candidate_detect_img = batch
    
    one_result_id = {}
    one_result_value = {}

    for cast_id in movie_cast_id:
        one_result_id[cast_id] = []
        one_result_value[cast_id] = []
    
    if cuda:
        # cast_img = Variable(movie_cast_img).cuda()
        candidate_img = Variable(movie_candidate_img)
        
    
    with torch.no_grad():
        ### Detect cast image ###
        cast_face = []
        cast_det_img = []

        # with open("feature_save/test2/cast/face/"+movie_name+".pickle", 'rb') as f:
        #     det_imgs = pickle.load(f)

        for i in range(len(movie_cast_detect_img)):
            det_img = detect(movie_cast_detect_img[i], 'cast')
            det_img = det_imgs[i]
            cast_det_img.append(det_img)
            cast_face.append(det_img.unsqueeze(0))
        cast_face = torch.cat(cast_face,dim=0)

        with open("feature_save/test4/cast/face/"+movie_name+".pickle", 'wb') as f:
            pickle.dump(cast_det_img, f)

        ### Extract cast face image feature ###
        # _, cast_feature = model_face(cast_face.cuda())
        # _, cast_feature = model_face(cast_face.cuda())
        cast_feature = extract_feature(cast_face.cuda(), model_face)
        
        cast_feature_np = cast_feature.cpu().numpy()
        np.save("feature_save/test4/cast/face/"+movie_name, cast_feature_np)
        # print(cast_feature.size())
        # cast_feature_np = np.load("feature_save/test2/cast/face/"+movie_name+".npy")
        # cast_feature = torch.from_numpy(cast_feature_np)

        ### Extract candidate body image feature ###
        candidate_body_feature = []
        for idx in range(candidate_img.size(0)):
            # _, one_candidate_feature = model_body(candidate_img[idx].cuda().unsqueeze(0))
            one_candidate_feature = extract_body_feature(candidate_img[idx].cuda().unsqueeze(0), model_body)
            
            # print(one_candidate_feature.size())
            # exit()
            candidate_body_feature.append(one_candidate_feature)
        candidate_body_feature = torch.cat(candidate_body_feature, dim=0)
        candidate_body_feature_np = candidate_body_feature.cpu().numpy()
        np.save("feature_save/test4/candidate/body/"+movie_name, candidate_body_feature_np)
        # candidate_body_feature_np = np.load("feature_save/test2/candidate/body/"+movie_name+".npy")
        # candidate_body_feature = torch.from_numpy(candidate_body_feature_np)

        ### Extract candidate face image feature ###
        # with open("feature_save/test2/candidate/face/"+movie_name+".pickle", 'rb') as f:
        #     det_imgs = pickle.load(f)

        candidate_face_feature = []
        candidate_det_img = []
        for i in range(len(movie_candidate_detect_img)):
            det_img = detect(movie_candidate_detect_img[i], 'candidate')
            # det_img = det_imgs[i]
            candidate_det_img.append(det_img)
            if det_img is not None:
                # _, one_candidate_face_feature = model_face(det_img.unsqueeze(0).cuda())
                # _, one_candidate_face_feature = model_face(det_img.unsqueeze(0).cuda())
                one_candidate_face_feature = extract_feature(det_img.unsqueeze(0).cuda(), model_face)
                candidate_face_feature.append(one_candidate_face_feature)
                # print(one_candidate_face_feature.size())
            else:
                candidate_face_feature.append(torch.zeros((1,512)).cuda())
        
        with open("feature_save/test4/candidate/face/"+movie_name+".pickle", 'wb') as f:
            pickle.dump(candidate_det_img, f)

        # exit()
        candidate_face_feature = torch.cat(candidate_face_feature, dim=0)
        candidate_face_feature_np = candidate_face_feature.cpu().numpy()
        np.save("feature_save/test4/candidate/face/"+movie_name, candidate_face_feature_np)
        # candidate_face_feature_np = np.load("feature_save/test2/candidate/face/"+movie_name+".npy")
        # candidate_face_feature = torch.from_numpy(candidate_face_feature_np)

        assert candidate_body_feature.size(0) == candidate_face_feature.size(0)
        # print(candidate_body_feature.size())
        # print(candidate_face_feature.size())
        # exit()

        ### Calculate cosine similarity between each cast feature and all candidate img feature
        ct_affmat = []
        for cast_idx in range(cast_feature.size(0)):
            cast_rep_feature = cast_feature[cast_idx].unsqueeze(0).repeat(candidate_face_feature.size(0),1)
            similarity = cos_similarity(cast_rep_feature, candidate_face_feature).unsqueeze(0)
            ct_affmat.append(similarity)

        tt_affmat = []
        ### Calculate cosine similarity between each candidate feature and all candidate img feature
        for cand_idx in range(candidate_body_feature.size(0)):
            cand_feature = candidate_body_feature[cand_idx].unsqueeze(0).repeat(candidate_body_feature.size(0),1)
            similarity = cos_similarity(cand_feature, candidate_body_feature).unsqueeze(0)
            tt_affmat.append(similarity)
            
        ct_affmat = torch.cat(ct_affmat, dim=0)
        tt_affmat = torch.cat(tt_affmat, dim=0)
        
        ### Run PPCC algorithm ###
        tnum = candidate_body_feature.size(0)
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
with open("output/face_ppcc/test3.json", 'w') as f:
    json.dump(result_id, f)
      
import csv
submission = open(os.path.join(out_file, "submit_test_face_propagation3.csv"), "w+") # "./{}.csv".format(target_dataset_name)
s = csv.writer(submission,delimiter=',',lineterminator='\n')
s.writerow(["Id","Rank"])

for mid, candidate in result_id.items():
    # print(candidate)
    s.writerow([mid, ' '.join(candidate)])

submission.close()




    

