
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
        
        ### Calculate cosine similarity between each candidate feature and all cast img feature
        for idx in range(candidate_feature.size(0)):
            cand_feature = candidate_feature[idx].unsqueeze(0).repeat(cast_feature.size(0),1)
            similarity = cos_similarity(cast_feature, cand_feature)

            # max_value, max_index = torch.max(similarity, 0) ### get the max value from the similarity score
            # max_value = max_value.cpu().data.item()
            # max_index = max_index.cpu().data.item()

            # similarity_score, similarity_score_index = similarity.sort(descending=True)

            # for index in range(len(similarity_score_index)):
            #     if similarity_score[index].item()>0.001:
            #         one_result_id[movie_cast_id[similarity_score_index[index]]].append(movie_candidate_id[idx])
            #         one_result_value[movie_cast_id[similarity_score_index[index]]].append(similarity_score[index].item())

            for index in range(len(similarity)):
                if similarity[index] > 0.1:
                    one_result_id[movie_cast_id[index]].append(movie_candidate_id[idx])
                    one_result_value[movie_cast_id[index]].append(similarity[index])
                
            # if max_value > 0.001: ### a threshold (need to change)
            #     # predict_id.append(movie_cast_id[max_index])
            #     one_result_id[movie_cast_id[max_index]].append(movie_candidate_id[idx])
            #     one_result_value[movie_cast_id[max_index]].append(max_value)
        
        ### Post-sort the candidate for each cast id, according to their similarity value ###
        for k, _ in one_result_id.items():
            one_result_id[k] = np.array(one_result_id[k])
            one_result_value[k] = np.array(one_result_value[k])

            out_arr = np.argsort(one_result_value[k])[::-1]
            one_result_id[k] = one_result_id[k][out_arr] 
            one_result_value[k] = one_result_value[k][out_arr] 
        
        result_id.update(one_result_id)
        result_value.update(one_result_value)



            
import csv
submission = open(os.path.join(out_file, "submit_val.csv"), "w+") # "./{}.csv".format(target_dataset_name)
s = csv.writer(submission,delimiter=',',lineterminator='\n')
s.writerow(["Id","Rank"])

for mid, candidate in result_id.items():
    s.writerow([mid, ' '.join(candidate)])

submission.close()




    

