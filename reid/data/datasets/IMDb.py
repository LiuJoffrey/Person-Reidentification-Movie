# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re
import os
import json

import os.path as osp

from .bases import BaseImageDataset


class IMDb(BaseImageDataset):
    dataset_dir = 'IMDb'

    def __init__(self, root='/media/cnrgntu11/left_PC/r06942141/DLCV/final/dataset', verbose=True, **kwargs):
        super(IMDb, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.val_dir = osp.join(self.dataset_dir, 'val')
        self.test_dir = osp.join(self.dataset_dir, 'test')

        self._check_before_run()

        train = self._process_train_dir(self.train_dir)
        val = self._process_val_dir(self.val_dir)
        test = self._process_test_dir(self.test_dir)


        if verbose:
            print("=> IMDb loaded")
            #self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.val = val
        self.test = test

        self.num_train_imgs, self.num_train_labels = self.get_IMDb_imagedata_info(self.train, training=True)
        # print(self.num_train_imgs, self.num_train_labels)
        # exit()
        # self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        # self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.val_dir):
            raise RuntimeError("'{}' is not available".format(self.val_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))


    # ======== Process training data =========
    def _process_train_dir(self, dir_path):

        folderlist = osp.join(dir_path)
        video_folder = sorted(os.listdir(folderlist))

        all_imgs_path = []
        all_labels = {}

        for video in video_folder:
            # ======= Process candidate  images =======
            candidate_json = osp.join(folderlist, video, 'candidate.json')
            with open(candidate_json, 'r') as f:
                candidate_file = json.load(f)

                for img, label in candidate_file.items():
                    if label == 'others': continue
                    else:
                        all_imgs_path.append((osp.join(self.dataset_dir, img), label))

            # ======= Process cast images =======
            cast_json = osp.join(folderlist, video, 'cast.json')
            with open(cast_json, 'r') as f:
                cast_file = json.load(f)

                for img, label in cast_file.items():
                    if label == 'others': continue
                    else:
                        all_imgs_path.append((osp.join(self.dataset_dir, img), label))

        return all_imgs_path

    # ======== Process validation data =========
    def _process_val_dir(self, dir_path):

        folderlist = osp.join(dir_path)
        video_folder = sorted(os.listdir(folderlist))

        movie_img_path_with_label = {}

        for video in video_folder:
            movie_cast_candidate = {}

            # ======= Process cast images =======   
            cast = []
            cast_json = osp.join(folderlist, video, 'cast.json')
            with open(cast_json, 'r') as f:
                cast_file = json.load(f)
                for img, label in cast_file.items():
                    if label == 'others': continue
                    else:
                        cast.append((osp.join(self.dataset_dir, img), label))
            movie_cast_candidate['cast'] = cast

            # ======= Process candidate  images =======
            candidate = []
            candidate_json = osp.join(folderlist, video, 'candidate.json')
            with open(candidate_json, 'r') as f:
                candidate_file = json.load(f)
                for img, label in candidate_file.items():
                    if label == 'others': continue
                    else:
                        candidate.append((osp.join(self.dataset_dir, img), label))
            movie_cast_candidate['candidate'] = candidate

            movie_img_path_with_label[video] = movie_cast_candidate
        
        return movie_img_path_with_label

    # ======== Process testing data =========
    def _process_test_dir(self, dir_path):

        folderlist = osp.join(dir_path)
        video_folder = sorted(os.listdir(folderlist))

        movie_img_path_with_label = {}

        for video in video_folder:
            movie_cast_candidate = {}

            # ======= Process cast images =======
            cast_path = osp.join(folderlist, video, 'cast')
            cast_img_dir = sorted(os.listdir(cast_path))
            cast = []
            for cast_img in cast_img_dir:
                cast_img_path = osp.join(cast_path, cast_img)
                cast.append((cast_img_path, cast_img[:-4]))
            movie_cast_candidate['cast'] = cast

            # ======= Process candidate  images =======
            candidate_path = osp.join(folderlist, video, 'candidates')
            candidate_img_dir = sorted(os.listdir(candidate_path))
            candidate = []
            for candidate_img in candidate_img_dir:
                candidate_img_path = osp.join(candidate_path, candidate_img)
                candidate.append((candidate_img_path, candidate_img[:-4]))
            movie_cast_candidate['candidate'] = candidate

            movie_img_path_with_label[video] = movie_cast_candidate
           
        return movie_img_path_with_label
