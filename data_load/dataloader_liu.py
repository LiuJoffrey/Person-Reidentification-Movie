from PIL import Image as io
import os
import numpy as np
import pandas as pd
import glob
import random
import matplotlib.pyplot as plt
import skvideo.io
import skimage.transform
import skimage.exposure
import cv2
import json
from random import randint

import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


class dl_bing(Dataset):
    def __init__(self,path='./Data',train=True,transform=None, mode='body'):    
         
        self.path = path
        self.train = train
        if mode == 'body':
            if train:
                folderlist=os.path.join(path,'train')
            else:
                folderlist=os.path.join(path,'val')
        elif mode == 'face':
            if train:
                folderlist=os.path.join(path,'train_face2')
            else:
                folderlist=os.path.join(path,'val_face2')

        self.video_folder = sorted(os.listdir(folderlist))

        allimg = None
        All_img_path_with_label = []
        All_img_path_with_others = []
        all_label = {}

        for f in self.video_folder:
            candidate_json = os.path.join(folderlist,f,'candidate.json')
            with open(candidate_json, 'r') as f_:
                candidate_file = json.load(f_)
                for k, l in candidate_file.items():
                    if l == 'others':
                        # All_img_path_with_others.append((os.path.join(self.path,k),l))
                        continue
                    else:
                        if os.path.isfile(os.path.join(self.path,k)):
                            All_img_path_with_label.append((os.path.join(self.path,k),l))

                    if l not in all_label:
                        all_label[l] = 1
                    else:
                        all_label[l] += 1

            cast_json = os.path.join(folderlist,f,'cast.json')
            with open(cast_json, 'r') as f_:
                cast_file = json.load(f_)
                for k, l in cast_file.items():
                    if l == 'others':
                        # All_img_path_with_others.append((os.path.join(self.path,k),l))
                        continue
                    else:
                        if os.path.isfile(os.path.join(self.path,k)):
                            All_img_path_with_label.append((os.path.join(self.path,k),l))

                    if l not in all_label:
                        all_label[l] = 1
                    else:
                        all_label[l] += 1
       
        self.All_img_path_with_label = All_img_path_with_label
        self.All_img_path_with_others = All_img_path_with_others
        self.all_label = all_label

        self.label = {}
        count = 0
        for k, v in self.all_label.items():
            self.label[k] = count
            count += 1

        self.transform = transform
        self.len=len(self.All_img_path_with_label)
        self.other_len = len(self.All_img_path_with_others)
        
    def __getitem__(self, index):
        
        img_path, label = self.All_img_path_with_label[index]
        # other_index = randint(0, self.other_len-1)
        # other_img_path, other = self.All_img_path_with_others[other_index]
        img = cv2.imread(img_path)

        # if self.train:
            # img = self.random_flip(img)
            # img = self.random_blur(img)
            # img = self.random_bright(img)
            # # img = self.random_hue(img)
            # img = self.random_saturation(img)
            # img = self.random_grayscale(img)
            # img = self.random_noise(img)

        img = self.BGR2RGB(img)
        # other_img = self.BGR2RGB(cv2.imread(other_img_path))

        # img = cv2.resize(img,(448,448))
        # other_img = cv2.resize(other_img,(448,448))

        if self.transform is not None:
            img = self.transform(img)
            # other_img = self.transform(other_img)
        
        # return [(img, self.label[label]), (other_img, self.label[other])]
        return img, self.label[label]

    
    def collate_fn(self, datas):

        all_img = []
        all_label = []

        for i in range(len(datas)):
            # cat a img with a other_img 
            all_img.append(datas[i][0][0].unsqueeze(0))
            all_img.append(datas[i][1][0].unsqueeze(0))
            # cat a img label with a other_img label
            all_label.append(datas[i][0][1])
            all_label.append(datas[i][1][1])

        return torch.cat(all_img, dim=0), torch.tensor(all_label)
    
    def __len__(self):
        return self.len 
    
    def BGR2RGB(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    def BGR2HSV(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    def HSV2BGR(self,img):
        return cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
    def BGR2GRA(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    def random_flip(self, img):
        #cv2.imshow("r",img)
        if random.random() > 0.5:
            h,w,c = img.shape
            img = img[:,::-1, :]

        return img 

    def random_blur(self, img):
        if random.random() > 0.5:
            img = cv2.GaussianBlur(img,(5,5),0)
        return img

    def random_bright(self, img):
        
        if random.random() > 0.5:
            value = int(random.uniform(-10,10))
            hsv = self.BGR2HSV(img)
            #hsv = hsv.astype('float32')
            h, s, v = cv2.split(hsv)
            v = v.astype('float32')

            v = v+value
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            final_hsv = cv2.merge((h, s, v))
            #final_hsv = final_hsv.astype('uint8')
            img = self.HSV2BGR(final_hsv)
        
        return img

    def random_hue(self, img):
        if random.random()>0.5:
            hsv = self.BGR2HSV(img)
            #hsv = hsv.astype('float32')
            h, s, v = cv2.split(hsv)
            value = random.uniform(0.5,1.2)
            h = h.astype('float32')
            h = h*value
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            final_hsv = cv2.merge((h,s,v))
            #final_hsv = final_hsv.astype('uint8')
            img = self.HSV2BGR(final_hsv)
            
        return img

    def random_saturation(self, img):
        if random.random()>0.5:
            hsv = self.BGR2HSV(img)
            #hsv = hsv.astype('float32')
            h, s, v = cv2.split(hsv)
            value = random.uniform(0.5,1.2)
            s = s.astype('float32')
            s = s*value
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            final_hsv = cv2.merge((h,s,v))
            #final_hsv = final_hsv.astype('uint8')
            img = self.HSV2BGR(final_hsv)
            
        return img

    def random_grayscale(self, img):
        if random.random()>0.5:
            img = self.BGR2GRA(img)
            img = np.expand_dims(img, 2)
            img = np.concatenate((img,img,img), axis=2)
            return img
        return img

    def random_noise(self, img):
        if random.random()>0.5:
            h, w, c = img.shape
            mean = 0
            var = 3
            sigma = var**0.5
            gaussian = np.random.normal(mean, sigma, (h, w))
            noisy_image = np.zeros(img.shape, np.float32)
            noisy_image[:, :, 0] = img[:, :, 0] + gaussian
            noisy_image[:, :, 1] = img[:, :, 1] + gaussian
            noisy_image[:, :, 2] = img[:, :, 2] + gaussian
            cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
            noisy_image = noisy_image.astype(np.uint8)
            return noisy_image

        if random.random()>0.5:
            output = np.zeros(img.shape,np.uint8)
            thres = 1 - 0.05 
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    rdn = random.random()
                    if rdn < 0.05:
                        output[i][j] = 0
                    elif rdn > 0.99:
                        output[i][j] = 255
                    else:
                        output[i][j] = img[i][j]
            return output

        return img

class dl_val_dataset(Dataset):
    def __init__(self,path='./Data',train=False,transform=None, mode='body'):        
        self.path = path
        if mode == 'body':
            folderlist=os.path.join(path,'val')
        elif mode == 'face':
            folderlist=os.path.join(path,'val_face')
        self.video_folder = sorted(os.listdir(folderlist))
        
        allimg = None
        All_img_path_with_label = []
        All_img_path_with_others = []
        all_label = {}
        
        self.movie_img_path_with_label = {} # To store all movies in the validation set
        for f in self.video_folder:
            candidate_json = os.path.join(folderlist,f,'candidate.json')
            cast_json = os.path.join(folderlist,f,'cast.json')
            ### To store the img path and img label for both cast and candidate ###
            one_movie_cast_candidate = {}
            cast = []
            with open(cast_json, 'r') as f_:
                cast_file = json.load(f_)
                for k, l in cast_file.items():
                    if os.path.isfile(os.path.join(self.path,k)):
                        cast.append((os.path.join(self.path,k),l))
            one_movie_cast_candidate["cast"] = cast
            
            candidate = []
            with open(candidate_json, 'r') as f_:
                candidate_file = json.load(f_)
                for k, l in candidate_file.items():
                    if os.path.isfile(os.path.join(self.path,k)):
                        candidate.append((os.path.join(self.path,k),l))
            one_movie_cast_candidate["candidate"] = candidate
            self.movie_img_path_with_label[f] = one_movie_cast_candidate
            
        self.transform = transform

        
    def __getitem__(self, index):
        
        movie_name = self.video_folder[index]
        movie_info = self.movie_img_path_with_label[movie_name]

        cast_data = movie_info["cast"]
        candidate_data = movie_info["candidate"]

        movie_cast_img = []
        movie_cast_label = []

        for i in range(len(cast_data)):
            cast_img_path = cast_data[i][0]
            cast_label = cast_data[i][1]
            cast_img = cv2.imread(cast_img_path)
            cast_img = self.BGR2RGB(cast_img)

            if self.transform is not None:
                cast_img = self.transform(cast_img)
            
            movie_cast_img.append(cast_img.unsqueeze(0))
            movie_cast_label.append(cast_label)

        movie_cast_img = torch.cat(movie_cast_img, dim=0)


        movie_candidate_img = []
        movie_candidate_label = []

        for i in range(len(candidate_data)):
            candidate_img_path = candidate_data[i][0]
            candidate_label = candidate_data[i][1]
            candidate_img = self.BGR2RGB(cv2.imread(candidate_img_path))

            if self.transform is not None:
                candidate_img = self.transform(candidate_img)
            
            movie_candidate_img.append(candidate_img.unsqueeze(0))
            movie_candidate_label.append(candidate_label)

        movie_candidate_img = torch.cat(movie_candidate_img, dim=0)
        
        return movie_cast_img, movie_cast_label, movie_candidate_img, movie_candidate_label

    
    def collate_fn(self, datas):
        movie_cast_img, movie_cast_label, movie_candidate_img, movie_candidate_label = datas[0]
        return movie_cast_img, movie_cast_label, movie_candidate_img, movie_candidate_label

    def __len__(self):
        return len(self.video_folder) 
    
    def BGR2RGB(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
class dl_test_dataset(Dataset):
    def __init__(self,path='./Data',train=False,transform=None):        
        self.path = path
       
        folderlist=os.path.join(path,'test')
        # folderlist=os.path.join(path,'val')
        self.video_folder = sorted(os.listdir(folderlist))
        self.video_folder = self.video_folder[:]
        # self.video_folder = [self.video_folder[4], self.video_folder[7],self.video_folder[18],self.video_folder[21]]

        self.movie_img_path_with_id = {} # To store all movies in the test set
        for f in self.video_folder:
            
            candidate_path = os.path.join(folderlist,f,'candidates')
            cast_path = os.path.join(folderlist,f,'cast')

            candidate_img_path = sorted(os.listdir(candidate_path))
            cast_img_path = sorted(os.listdir(cast_path))

            ### To store the img path and img id for both cast and candidate ###
            one_movie_cast_candidate = {}
            cast = []
            for f_ in cast_img_path:
                cast_img = os.path.join(cast_path,f_)
                cast.append((cast_img, f_.split('.')[0]))
    
            one_movie_cast_candidate["cast"] = cast

            
            candidate = []
            for f_ in candidate_img_path:
                candidate_img = os.path.join(candidate_path,f_)
                candidate.append((candidate_img, f_.split('.')[0]))
                
            one_movie_cast_candidate["candidate"] = candidate
            self.movie_img_path_with_id[f] = one_movie_cast_candidate
            
        self.transform = transform

        self.transform2 = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((384,128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        
    def __getitem__(self, index):

        
        
        movie_name = self.video_folder[index]
        movie_info = self.movie_img_path_with_id[movie_name]

        cast_data = movie_info["cast"]
        candidate_data = movie_info["candidate"]

        if_gray_img = cv2.imread(candidate_data[0][0])
        
        if np.all(if_gray_img[:,:,0]==if_gray_img[:,:,1]) and np.all(if_gray_img[:,:,1]==if_gray_img[:,:,2]):
            is_gray = True
        else:
            is_gray = False

        movie_cast_img = []
        movie_cast_img2 = []
        movie_cast_detect_img = []
        movie_cast_id = []

        for i in range(len(cast_data)):
            cast_img_path = cast_data[i][0]
            cast_id = cast_data[i][1]
            
            ori_img = cv2.imread(cast_img_path)

            if is_gray:
                ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
                ori_img = np.stack([ori_img, ori_img, ori_img], axis=-1)
                
            



            # detect_img = detect(ori_img, 'cast')
            # if detect_img is not None:
            #     movie_cast_detect_img.append(detect_img.unsqueeze(0))

            movie_cast_detect_img.append(ori_img)

            cast_img = self.BGR2RGB(ori_img)

            if self.transform is not None:
                cast_img = self.transform(cast_img)
                cast_img2 = self.transform2(cast_img)

                
            
            movie_cast_img.append(cast_img.unsqueeze(0))
            movie_cast_img2.append(cast_img2.unsqueeze(0))
            movie_cast_id.append(cast_id)

        movie_cast_img = torch.cat(movie_cast_img, dim=0)
        movie_cast_img2 = torch.cat(movie_cast_img2, dim=0)
       

        movie_candidate_img = []
        movie_candidate_img2 = []
        movie_candidate_detect_img = []
        movie_candidate_id = []

        for i in range(len(candidate_data)):
            candidate_img_path = candidate_data[i][0]
            candidate_id = candidate_data[i][1]

            ori_img = cv2.imread(candidate_img_path)

            # detect_img = detect(ori_img, 'candidate')
            # if detect_img is not None:
            #     movie_candidate_detect_img.append(detect_img)
            # else:
            #     movie_candidate_detect_img.append(None)
            movie_candidate_detect_img.append(ori_img)

            candidate_img = self.BGR2RGB(ori_img)

            if self.transform is not None:
                candidate_img = self.transform(candidate_img)
                candidate_img2 = self.transform2(candidate_img)
            
            movie_candidate_img.append(candidate_img.unsqueeze(0))
            movie_candidate_img2.append(candidate_img2.unsqueeze(0))
            movie_candidate_id.append(candidate_id)

        movie_candidate_img = torch.cat(movie_candidate_img, dim=0)
        movie_candidate_img2 = torch.cat(movie_candidate_img2, dim=0)
        
        return movie_cast_img2, movie_candidate_img2, movie_cast_img, movie_cast_id, movie_candidate_img, movie_candidate_id, movie_name, movie_cast_detect_img, movie_candidate_detect_img

    
    def collate_fn(self, datas):

        # movie_cast_img, movie_cast_id, movie_candidate_img, movie_candidate_id, movie_name = datas[0]
        movie_cast_img2, movie_candidate_img2, movie_cast_img, movie_cast_id, movie_candidate_img, movie_candidate_id, movie_name, movie_cast_detect_img, movie_candidate_detect_img = datas[0]

        # return movie_cast_img, movie_cast_id, movie_candidate_img, movie_candidate_id, movie_name
        return movie_cast_img2, movie_candidate_img2, movie_cast_img, movie_cast_id, movie_candidate_img, movie_candidate_id, movie_name, movie_cast_detect_img, movie_candidate_detect_img 

    def __len__(self):
        return len(self.video_folder) 
    
    def BGR2RGB(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

def test():
    print("Hello")

    train_path = "../../train"
    valid_path = "../../valid"

    transform = transforms.Compose([
                            transforms.ToPILImage(),
                            # transforms.Pad((0,40), fill=0, padding_mode='constant'),
                            transforms.Resize((227,227)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
    # return transform(image)

    path = "../../"

    # train_data = dl_bing(path, train=True,transform=transform)
    # dataload = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False) # , collate_fn=train_data.collate_fn
    # for i, (img, label) in enumerate(dataload):
    #     print(img.size())
    #     print(label)
    #     exit()


    train_data = dl_test_dataset(path, train=False,transform=transform)
    
    dataload = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False, collate_fn=train_data.collate_fn) # 
    for i, batch in enumerate(dataload):
        
        movie_cast_img, movie_cast_id, movie_candidate_img, movie_candidate_id, movie_name = batch

        print(movie_cast_img.size())
        print(len(movie_cast_id))
        print(movie_candidate_img.size())
        print(len(movie_candidate_id))
        exit()

    # train_data = dl_val_dataset(path, train=False,transform=transform)

    # dataload = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False, collate_fn=train_data.collate_fn) # 
    # for i, batch in enumerate(dataload):
        
    #     movie_cast_img, movie_cast_label, movie_candidate_img, movie_candidate_label = batch

    #     print(movie_cast_img.size())
    #     print(len(movie_cast_label))
    #     print(movie_candidate_img.size())
    #     print(len(movie_candidate_label))
    #     exit()




if __name__ == "__main__":
    test()
