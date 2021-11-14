
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
import new_rnn.model_liu_v2 as bing_model

# from detect import detect
from detect_big import detect
# from detect_all import detect

def loop_ccpp(ct_similarity, tt_mask2):
    result = run_ccpp(ct_similarity.cpu().numpy(), tt_mask2.cpu().numpy(), 0, init_fratio=0.5)
    result = 0.5*result + 0.5*ct_similarity.cpu().numpy()
    result = run_ccpp(result, tt_mask2.cpu().numpy(), 0, init_fratio=0.8)
    result = 0.5*result + 0.5*ct_similarity.cpu().numpy()
    result = run_ccpp(result, tt_mask2.cpu().numpy(), 0, init_fratio=0.9)
    result = 0.5*result + 0.5*ct_similarity.cpu().numpy()
    result = run_ccpp(result, tt_mask2.cpu().numpy(), 0, init_fratio=1.0)
    return result


def run_ccpp(ct_affmat, tt_affmat, gpu_id, init_fratio=0.5):
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
        result = gpu_ccpp(W, Y0, gpu_id=gpu_id, init_fratio=init_fratio)
    
    return result

def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    max_x = torch.max(x)
    min_x = torch.min(x)
    x = (x - min_x) / (max_x - min_x)
    #x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def get_cand_body_feature(candidate_img, model_body):
    candidate_body_feature = []
    for idx in range(candidate_img.size(0)):
        one_candidate_feature = extract_body_feature(candidate_img[idx].cuda().unsqueeze(0), model_body)
        candidate_body_feature.append(one_candidate_feature)
    candidate_body_feature = torch.cat(candidate_body_feature, dim=0)
    return candidate_body_feature

def get_body_ct_affmat(cast_body_feature, candidate_body_feature):
    body_ct_affmat = []
    for cast_idx in range(cast_body_feature.size(0)):
        cast_rep_feature = cast_body_feature[cast_idx].unsqueeze(0).repeat(candidate_body_feature.size(0),1)
        similarity = cos_similarity(cast_rep_feature, candidate_body_feature).unsqueeze(0)
        body_ct_affmat.append(similarity)
    body_ct_affmat = torch.cat(body_ct_affmat, dim=0)

    body_ct_eu_mat = euclidean_dist(cast_body_feature, candidate_body_feature)
    body_ct_eu_mat = 1-normalize(body_ct_eu_mat)

    body_ct_affmat_std = torch.std(body_ct_affmat)
    body_ct_affmat_mean = torch.mean(body_ct_affmat)
    body_ct_eu_mat_std = torch.std(body_ct_eu_mat)
    body_ct_eu_mat_mean = torch.mean(body_ct_eu_mat)
    body_ct_eu_mat = body_ct_eu_mat*(body_ct_affmat_std/body_ct_eu_mat_std) + (body_ct_affmat_mean-body_ct_eu_mat_mean)
    body_ct_similarity = (body_ct_eu_mat+body_ct_affmat)/2
    return body_ct_similarity

def get_body_tt_affmat(candidate_body_feature):
    tt_affmat = []
        ### Calculate cosine similarity between each candidate feature and all candidate img feature
    for cand_idx in range(candidate_body_feature.size(0)):
        cand_feature = candidate_body_feature[cand_idx].unsqueeze(0).repeat(candidate_body_feature.size(0),1)
        similarity = cos_similarity(cand_feature, candidate_body_feature).unsqueeze(0)
        tt_affmat.append(similarity)
        
    tt_affmat = torch.cat(tt_affmat, dim=0)
    tt_eu_mat = euclidean_dist(candidate_body_feature, candidate_body_feature)
    tt_eu_mat = 1-normalize(tt_eu_mat)

    tt_affmat_std = torch.std(tt_affmat)
    tt_affmat_mean = torch.mean(tt_affmat)
    tt_eu_mat_std = torch.std(tt_eu_mat)
    tt_eu_mat_mean = torch.mean(tt_eu_mat)
    tt_eu_mat = tt_eu_mat*(tt_affmat_std/tt_eu_mat_std) + (tt_affmat_mean-tt_eu_mat_mean)
    tt_similarity = (tt_eu_mat+tt_affmat)/2
    return tt_similarity

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

model_body_50 = r_net50(pretrained=False)
# model_body_101 = r_net101(pretrained=False)

model_body_RNET = bing_model.r_net50()
# model_body_QKV = bing_model.QKV(max_c=11)

def load_checkpoint(checkpoint_path, model,device):
    state = torch.load(checkpoint_path,map_location=device)
    model.load_state_dict(state['state_dict'])
    print('model loaded from %s' % checkpoint_path)

load_checkpoint('./new_rnn/RNET.pth', model_body_RNET, 'cuda')
# load_checkpoint('./models/NEW/QKV.pth', model_body_QKV, 'cuda')
exit()
# model_body_unsuper = r_net50(pretrained=False)

# model_body = r_net50_512(pretrained=False)
# model_face = r_net50()
model_face = IR_50([112,112])
cuda = True
if cuda:
    # model_body_101.cuda()
    model_body_50.cuda()
    model_face.cuda()

# model_body.load_state_dict(torch.load(os.path.join(
#                         model_body_save_path, '6_0.20049910317398426_model.pth')))
# model_face.load_state_dict(torch.load(os.path.join(
#                         model_face_save_path, '2_0.15695755023525437_model.pth')))
model_face.load_state_dict(torch.load("./models/backbone_ir50_ms1m_epoch120.pth"))
model_body_50.load_state_dict(torch.load(
            "/media/cnrg-ntu2/HDD1TB/r07921052/DLCV/final/challenge2/final-bienaola-master/checkpoint/body/6_0.20049910317398426_model.pth"))
# model_body_101.load_state_dict(torch.load(
#             "/media/cnrg-ntu2/HDD1TB/r07921052/DLCV/final/challenge2/final-bienaola-master/checkpoint/body/33_0.2288336062803816_model.pth"))

# model_body_101.eval()
model_body_50.eval()
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
    pids = movie_cast_id
    # model_body.load_state_dict(torch.load(
    #         "/media/cnrg-ntu2/HDD1TB/r07921052/DLCV/final/challenge2/final-bienaola-unsuper/checkpoint/unsuper_face_cosine/"+movie_name+"_10_.pth"))
    # model_body_unsuper.load_state_dict(torch.load(
    #         "/media/cnrg-ntu2/HDD1TB/r07921052/DLCV/final/challenge2/final-bienaola-unsuper/checkpoint/unsuper_face_cosine/"+movie_name+"_10_recenter.pth"))
    # model_body.load_state_dict(torch.load(
    #         "/media/cnrg-ntu2/HDD1TB/r07921052/DLCV/final/challenge2/final-bienaola-master/checkpoint/body/6_0.20049910317398426_model.pth"))
    # model_body.load_state_dict(torch.load(
    #         "/media/cnrg-ntu2/HDD1TB/r07921052/DLCV/final/challenge2/final-bienaola-master/checkpoint/body/23_0.22158101328342303_model.pth"))
    
    one_result_id = {}
    one_result_value = {}

    for cast_id in movie_cast_id:
        one_result_id[cast_id] = []
        one_result_value[cast_id] = []
    
    if cuda:
        cast_img = Variable(movie_cast_img).cuda()
        candidate_img = Variable(movie_candidate_img)
        
    
    with torch.no_grad():
        ### Detect cast image ###
        cast_face = []
        cast_det_img = []

        #### Get cast face feature ###
        with open("feature_save/test6/cast/face/"+movie_name+".pickle", 'rb') as f:
            det_imgs = pickle.load(f)

        for cast_idx in range(len(movie_cast_detect_img)):
            # print(cast_idx)
            # if cast_idx !=6:
            #     continue
            
            # det_img = detect(movie_cast_detect_img[cast_idx], 'cast')
            
            det_img = det_imgs[cast_idx]
            cast_det_img.append(det_img)
            cast_face.append(det_img.unsqueeze(0))
        # exit()
        cast_face = torch.cat(cast_face,dim=0)
        cast_feature = extract_feature(cast_face.cuda(), model_face)
        
        # with open("feature_save/test7/cast/face/"+movie_name+".pickle", 'wb') as f:
        #     pickle.dump(cast_det_img, f)
        
        #### Get cast body feature ###
        cast_body_feature_50 = extract_body_feature(cast_img.cuda(), model_body_50)
        # cast_body_feature_101 = extract_body_feature(cast_img.cuda(), model_body_101)
        
        ### Extract candidate face image feature ###
        # with open("feature_save/test2/candidate/face/"+movie_name+".pickle", 'rb') as f:
        #     det_imgs = pickle.load(f)
        with open("feature_save/test6/candidate/face/"+movie_name+".pickle", 'rb') as f:
            det_imgs = pickle.load(f)

        candidate_face_feature = []
        candidate_det_img = []
        for cand_idx in range(len(movie_candidate_detect_img)):
            # print(gt[cand_idx])
            # det_img = detect(movie_candidate_detect_img[cand_idx], 'candidate')
            # continue
            det_img = det_imgs[cand_idx]
            candidate_det_img.append(det_img)

            if det_img is not None:
                one_candidate_face_feature = extract_feature(det_img.unsqueeze(0).cuda(), model_face)
                candidate_face_feature.append(one_candidate_face_feature)
                # print(one_candidate_face_feature.size())
            else:
                candidate_face_feature.append(torch.zeros((1,512)).cuda())
        # exit()
        candidate_face_feature = torch.cat(candidate_face_feature, dim=0)
        # with open("feature_save/test7/candidate/face/"+movie_name+".pickle", 'wb') as f:
        #     pickle.dump(candidate_det_img, f)
        # continue
        
        ### Extract candidate body image feature ###
        # candidate_body_feature = []
        # for idx in range(candidate_img.size(0)):
        #     one_candidate_feature = extract_body_feature(candidate_img[idx].cuda().unsqueeze(0), model_body)
        #     candidate_body_feature.append(one_candidate_feature)
        # candidate_body_feature = torch.cat(candidate_body_feature, dim=0)

        candidate_body_feature_50 = get_cand_body_feature(candidate_img, model_body_50)
        # candidate_body_feature_101 = get_cand_body_feature(candidate_img, model_body_101)


        ### Calculate cosine similarity between each cast feature and all candidate img feature
        ct_affmat = []
        for cast_idx in range(cast_feature.size(0)):
            cast_rep_feature = cast_feature[cast_idx].unsqueeze(0).repeat(candidate_face_feature.size(0),1)
            similarity = cos_similarity(cast_rep_feature, candidate_face_feature).unsqueeze(0)
            ct_affmat.append(similarity)

        ct_affmat = torch.cat(ct_affmat, dim=0)
        ct_eu_mat = euclidean_dist(cast_feature, candidate_face_feature)
        ct_eu_mat = 1-normalize(ct_eu_mat)

        ct_affmat_std = torch.std(ct_affmat)
        ct_affmat_mean = torch.mean(ct_affmat)
        ct_eu_mat_std = torch.std(ct_eu_mat)
        ct_eu_mat_mean = torch.mean(ct_eu_mat)
        ct_eu_mat = ct_eu_mat*(ct_affmat_std/ct_eu_mat_std) + (ct_affmat_mean-ct_eu_mat_mean)
        ct_similarity = (ct_eu_mat+ct_affmat)/2

        ### Calculate cosine similarity between each cast body feature and all candidate img body feature   
        body_ct_similarity_50_50 = get_body_ct_affmat(cast_body_feature_50, candidate_body_feature_50)
        # body_ct_similarity_50_101 = get_body_ct_affmat(cast_body_feature_50, candidate_body_feature_101)
        # body_ct_similarity_101_50 = get_body_ct_affmat(cast_body_feature_101, candidate_body_feature_50)
        # body_ct_similarity_101_101 = get_body_ct_affmat(cast_body_feature_101, candidate_body_feature_101)
        # affmat_std_50 = torch.std(body_ct_similarity_50_50)
        # affmat_mean_50 = torch.mean(body_ct_similarity_50_50)
        # mat_std_101 = torch.std(body_ct_similarity_101_101)
        # mat_mean_101 = torch.mean(body_ct_similarity_101_101)
        # body_ct_similarity_101_101 = body_ct_similarity_101_101*(affmat_std_50/mat_std_101) + (affmat_mean_50-mat_mean_101)

        # body_ct_similarity = (body_ct_similarity_50_50+body_ct_similarity_101_101)/2
        body_ct_similarity = (body_ct_similarity_50_50)/1
        
        
        ### Calculate cosine similarity between each candidate feature and all candidate img feature
        tt_similarity_50 = get_body_tt_affmat(candidate_body_feature_50)
        # tt_similarity_101 = get_body_tt_affmat(candidate_body_feature_101)
        # tt_similarity = (tt_similarity_50+tt_similarity_101)/2
        tt_similarity = (tt_similarity_50)/1


        # tt_affmat_std = torch.std(tt_similarity)
        # tt_affmat_mean = torch.mean(tt_similarity)
        # body_tt_mat_std = torch.std(body_ct_similarity)
        # body_tt_mat_mean = torch.mean(body_ct_similarity)
        # body_ct_similarity = body_ct_similarity*(tt_affmat_std/body_tt_mat_std) + (tt_affmat_mean-body_tt_mat_mean)

        tt_mask2=torch.zeros(tt_similarity.shape)
        window = 50
        for i in range(0,tt_similarity.shape[0]):
            if i+window <tt_similarity.shape[0]:
                tt_mask2[i:(i+window),i:(i+window)]=tt_similarity[i:i+window,i:i+window]
            else:
                tt_mask2[i:,i:]=tt_similarity[i:,i:]

        face_tt_affmat = []
        for cand_idx in range(candidate_face_feature.size(0)):
            cand_feature = candidate_face_feature[cand_idx].unsqueeze(0).repeat(candidate_face_feature.size(0),1)
            similarity = cos_similarity(cand_feature, candidate_face_feature).unsqueeze(0)
            face_tt_affmat.append(similarity)
        face_tt_affmat = torch.cat(face_tt_affmat, dim=0)
        face_tt_eu_mat = euclidean_dist(candidate_face_feature, candidate_face_feature)
        face_tt_eu_mat = 1-normalize(face_tt_eu_mat)

        face_tt_affmat_std = torch.std(face_tt_affmat)
        face_tt_affmat_mean = torch.mean(face_tt_affmat)
        face_tt_eu_mat_std = torch.std(face_tt_eu_mat)
        face_tt_eu_mat_mean = torch.mean(face_tt_eu_mat)
        face_tt_eu_mat = face_tt_eu_mat*(face_tt_affmat_std/face_tt_eu_mat_std) + (face_tt_affmat_mean-face_tt_eu_mat_mean)
        face_tt_similarity = (face_tt_eu_mat+face_tt_affmat)/2

        face_tt_mask2=torch.zeros(face_tt_similarity.shape)
        window = 50
        for i in range(0,face_tt_similarity.shape[0]):
            if i+window <face_tt_similarity.shape[0]:
                face_tt_mask2[i:(i+window),i:(i+window)]=face_tt_similarity[i:i+window,i:i+window]
            else:
                face_tt_mask2[i:,i:]=face_tt_similarity[i:,i:]

        face_result = loop_ccpp(ct_similarity, face_tt_mask2)

        ### ========= ###
        face_sort = np.argsort(-face_result, axis=-1)
        
        cast_dominate_result = {}
        cast_negative_result = {}
        for cast_id in range(len(face_sort)):
            cast_negative_result[movie_cast_id[cast_id]] = []

        new_cand_to_cast = []
        new_cand_to_cast_len = []
        cast_donimate_similarity = []

        cast_donimate_similarity_max_value = []
        
        # with open("/media/cnrg-ntu2/HDD1TB/r07921052/DLCV/final/challenge2/val/tt0209958/candidate.json", 'r') as f:
        #     gt = json.load(f)
        
        # gt = list(gt.values())

        for cast_idx in range(len(face_sort)):
            cast_dominate = []
            cast_dominate_result[movie_cast_id[cast_idx]] = []
            # print(face_sort[cast_idx])
            cast_donimate_similarity_max_value.append(max(face_result[cast_idx]))
            print(cast_donimate_similarity_max_value)

            cast_count = 0
            for candidate_idx in face_sort[cast_idx]:
                if face_result[cast_idx, candidate_idx] > 2.06:

                    cast_dominate.append(candidate_idx)
                    cast_dominate_result[movie_cast_id[cast_idx]].append(movie_candidate_id[candidate_idx])

                    for neg_cast_id in range(len(face_sort)):
                        if neg_cast_id == cast_idx:
                            continue
                        cast_negative_result[movie_cast_id[neg_cast_id]].append(movie_candidate_id[candidate_idx])
                    
                    # cast_donimate_similarity.append(tt_similarity[candidate_idx].unsqueeze(0))
                    cast_donimate_similarity.append(tt_mask2[candidate_idx].unsqueeze(0).cuda())
                    # cast_donimate_similarity.append(face_tt_affmat[candidate_idx].unsqueeze(0))

                    cast_count += 1
                    
                elif len(cast_dominate)==0:
                    cast_donimate_similarity.append(body_ct_similarity[cast_idx].unsqueeze(0))
                    # cast_donimate_similarity.append(ct_similarity[cast_idx].unsqueeze(0))
                    break
        

            new_cand_to_cast_len.append(len(cast_dominate))
        # exit()
        print(new_cand_to_cast_len)
        
        cast_donimate_similarity = torch.cat(cast_donimate_similarity, dim=0)

        start = 0
        cast_donimate_similarity_max = []
        for cast_idx, cand_len in enumerate(new_cand_to_cast_len):
            if cand_len!=0:
                same_cast = cast_donimate_similarity[start:start+cand_len]
                start += cand_len
                cast_donimate_similarity_max.append(torch.max(same_cast, 0)[0].unsqueeze(0))
            else:
                same_cast = cast_donimate_similarity[start:start+1]
                start = start+1
                cast_donimate_similarity_max.append(torch.max(same_cast, 0)[0].unsqueeze(0))

        cast_donimate_similarity_max = torch.cat(cast_donimate_similarity_max, dim=0)        

        ### Run PPCC algorithm ###
        # tnum = candidate_body_feature.size(0)

        # print(face_tt_similarity.size())
        # print(tt_similarity.size())

        # for cand_idx in range(len(movie_candidate_detect_img)):
        #     if det_imgs[cand_idx] is None:
        #         face_tt_similarity[cand_idx] = tt_similarity[cand_idx]
        # for cand_idi in range(face_tt_similarity.size(0)):
        #     for cand_idj in range(face_tt_similarity.size(1)):
        #         if det_imgs[cand_idi] is None or det_imgs[cand_idj] is None:
        #             face_tt_similarity[cand_idi, cand_idj] = tt_similarity[cand_idi, cand_idj]

        # face_tt_mask2=torch.zeros(face_tt_similarity.shape)
        # window = 50
        # for i in range(0,face_tt_similarity.shape[0]):
        #     if i+window <face_tt_similarity.shape[0]:
        #         face_tt_mask2[i:(i+window),i:(i+window)]=face_tt_similarity[i:i+window,i:i+window]
        #     else:
        #         face_tt_mask2[i:,i:]=face_tt_similarity[i:,i:]

        
        result_cast_ct = loop_ccpp(ct_similarity, tt_mask2)
        # result_cast_ct = loop_ccpp(ct_similarity, face_tt_mask2)


        result_candidate_ct = loop_ccpp(cast_donimate_similarity_max, tt_mask2)

        result = result_candidate_ct

        # result = (result_candidate_ct+result_candidate_ct)/2
        # result = result_candidate_ct



        for cast_idx in range(len(result_cast_ct)):
            if new_cand_to_cast_len[cast_idx] == 0 and cast_donimate_similarity_max_value[cast_idx] < 1.9: 
                result_cast_ct[cast_idx] = result_candidate_ct[cast_idx]

        result = result_cast_ct


        

        
        # for cast_idx in range(len(result)):
        #     if new_cand_to_cast_len[cast_idx] == 0 and cast_donimate_similarity_max_value[cast_idx] < 1.9: 
        #         continue
        #     for cand_idx in range(len(result[0])):


        




        # print(result_cast_ct[0])
        # print(result_candidate_ct[0])
        # result = np.zeros((result_cast_ct.shape[0], result_cast_ct.shape[1]))

        # for cast_idx in range(len(result)):
        #     for candidate_idx in range(len(result[0])):
        #         if det_imgs[i] is None:
        #             result[cast_idx, candidate_idx] = result_candidate_ct[cast_idx,candidate_idx]
        #         else:
        #             result[cast_idx, candidate_idx] = result_cast_ct[cast_idx,candidate_idx]
                    


        # for cand_idx in range(len(result[0])):
        #     print(result_cast_ct[0,cand_idx])
        #     print(result_candidate_ct[0,cand_idx])
        #     if det_imgs[i] is None:
        #         print("No face")
        #     else:
        #         print("Has face")
        #     print("=====")
        #     cv2.imshow("x", movie_candidate_detect_img[cand_idx])
        #     cv2.waitKey(0)

        # exit()

        for idx in range(len(result[0]))[:]:
            column = result[:, idx]
            mean = np.mean(column)
            std = np.std(column)
            max_id = np.argmax(column)
            max_value = np.max(column)


            if std < 0.05:
                for cast_idx in range(len(result)):
                    #if cast_idx != max_id:
                    result[cast_idx, idx] = -100


        

        ret_dict = affmat2retdict(result, pids)


        after_ppcc_result_idx = {}
        for cast_idx in range(len(ret_dict)):
            after_ppcc_result_idx[movie_cast_id[cast_idx]] = []
            for cand_idx in ret_dict[movie_cast_id[cast_idx]]:
                if result[cast_idx, cand_idx]>2.0:
                    after_ppcc_result_idx[movie_cast_id[cast_idx]].append(movie_candidate_id[cand_idx])
        
        # #### all negative ###
        # for c, (k,_) in enumerate(after_ppcc_result_idx.items()):
        #     for neg_k,_ in after_ppcc_result_idx.items():
        #         if neg_k!=k:
        #             cast_negative_result[k] += after_ppcc_result_idx[neg_k]
        #     cast_negative_result[k] = list(set(cast_negative_result[k]))



        print(after_ppcc_result_idx)
        ## Transfer candidate id index(int) to candidate id(str)
        
        one_result_id = {}
        for c, (k, _) in enumerate(ret_dict.items()):
            if new_cand_to_cast_len[c] == 0 and cast_donimate_similarity_max_value[c] < 1.9:
                print("Use all neg")
                all_neg = []
                for neg_k,_ in after_ppcc_result_idx.items():
                    if neg_k!=k:
                        all_neg += after_ppcc_result_idx[neg_k]
                
                cast_negative_result[k] += all_neg
                cast_negative_result[k] = list(set(cast_negative_result[k]))

                cast_candidate_index = ret_dict[k]
                candidate_index_to_id = []

                for donimate_id in cast_dominate_result[k]:
                    candidate_index_to_id.append(donimate_id)
                
                for cand_idx in cast_candidate_index:
                    if movie_candidate_id[cand_idx] in candidate_index_to_id or movie_candidate_id[cand_idx] in cast_negative_result[k]:
                        continue
                    
                    candidate_index_to_id.append(movie_candidate_id[cand_idx])  

                for candidate_id in cast_negative_result[k]:
                    candidate_index_to_id.append(candidate_id)
                one_result_id[k] = candidate_index_to_id

            else:
                cast_candidate_index = ret_dict[k]
                candidate_index_to_id = []

                for donimate_id in cast_dominate_result[k]:
                    candidate_index_to_id.append(donimate_id)

                for cand_idx in cast_candidate_index:
                    if movie_candidate_id[cand_idx] in candidate_index_to_id or movie_candidate_id[cand_idx] in cast_negative_result[k]:
                        continue
                    
                    candidate_index_to_id.append(movie_candidate_id[cand_idx])   

                for candidate_id in cast_negative_result[k]:
                    candidate_index_to_id.append(candidate_id)

                one_result_id[k] = candidate_index_to_id
        result_id.update(one_result_id)

        # one_result_id = {}
        # for c, (k, _) in enumerate(ret_dict.items()):
        #     cast_candidate_index = ret_dict[k]
        #     candidate_index_to_id = []

        #     for donimate_id in cast_dominate_result[k]:
        #         candidate_index_to_id.append(donimate_id)

        #     for cand_idx in cast_candidate_index:
        #         if movie_candidate_id[cand_idx] in candidate_index_to_id or movie_candidate_id[cand_idx] in cast_negative_result[k]:
        #             continue
                
        #         candidate_index_to_id.append(movie_candidate_id[cand_idx])   

        #     for candidate_id in cast_negative_result[k]:
        #         candidate_index_to_id.append(candidate_id)

        #     one_result_id[k] = candidate_index_to_id
        # result_id.update(one_result_id)


import json    
with open("output/face_ppcc/val_split_ensem.json", 'w') as f:
    json.dump(result_id, f)
      
import csv
submission = open(os.path.join(out_file, "submit_val_face_propagation_split_ensem.csv"), "w+") # "./{}.csv".format(target_dataset_name)
s = csv.writer(submission,delimiter=',',lineterminator='\n')
s.writerow(["Id","Rank"])

for mid, candidate in result_id.items():
    # print(candidate)
    s.writerow([mid, ' '.join(candidate)])

submission.close()




    

