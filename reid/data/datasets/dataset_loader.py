# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

        # ===== labels index dictionary =====
        self.labels = {}
        count = 0
        for i, (img_path, label) in enumerate(dataset):
            if label not in self.labels:
                self.labels[label] = count
                count += 1

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, label = self.dataset[index]
        img = read_image(img_path)


        if self.transform is not None:
            img = self.transform(img)

        return img, self.labels[label], img_path

class ValImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.movie_name = []
        for k, v in dataset.items():
            self.movie_name.append(k)

        self.transform = transform

    def __len__(self):
        return len(self.movie_name)

    def __getitem__(self, index):

        movie_name = self.movie_name[index]
        movie_info = self.dataset[movie_name]
        cast_data = movie_info['cast']
        candidate_data = movie_info['candidate']

        # ========= Process cast data ===========
        cast_imgs = []
        cast_labels = []
        for i in range(len(cast_data)):
            # ==== Load imgs & labels ====
            img = read_image(cast_data[i][0])
            cast_labels.append(cast_data[i][1])

            # ==== imgs process ====
            if self.transform is not None:
                img = self.transform(img)
            
            cast_imgs.append(img.unsqueeze(0))
        cast_imgs = torch.cat(cast_imgs, dim=0)

        # ========= Process candidate data ===========
        candidate_imgs = []
        candidate_labels = []
        for i in range(len(candidate_data)):
            # ==== Load imgs & labels ====
            img = read_image(candidate_data[i][0])
            candidate_labels.append(candidate_data[i][1])

            # ==== imgs process ====
            if self.transform is not None:
                img = self.transform(img)
            
            candidate_imgs.append(img.unsqueeze(0))
        candidate_imgs = torch.cat(candidate_imgs, dim=0)

        return cast_imgs, cast_labels, candidate_imgs, candidate_labels
