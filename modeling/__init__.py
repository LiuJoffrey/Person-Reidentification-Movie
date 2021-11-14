# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .baseline import Baseline
import torch


def build_model(cfg, num_classes):
    if cfg.MODEL.NAME == 'resnet50':
        model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH)
    return model

def build_model_test():
    return Baseline(940, 1, '/media/cnrg-ntu2/HDD1TB/r07921052/DLCV/final/challenge2/final-bienaola-master/modeling/31_6.910599997663869_model.pth')

def extract_body_feature(imgs, model, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), tta = True):
    
    flipped = torch.flip(imgs, [3]) 

    model.to(device)
    # extract features
    model.eval() # set to evaluation mode
    with torch.no_grad():
        if tta:
            emb_batch = model(imgs.to(device)) + model(flipped.to(device))
            features = l2_norm(emb_batch)
        else:
            features = l2_norm(model(ccropped.to(device)).cpu())

    return features

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output
    
