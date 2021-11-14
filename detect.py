from detection_utils import detect_faces, show_bboxes
from PIL import Image
import numpy as np
import os
import sys
import cv2
from torch.autograd import Variable
from shutil import copyfile
import json
import torchvision.transforms as transforms

transform = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Pad((0,40), fill=0, padding_mode='constant'),
            transforms.Resize((112,112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])

def detect(img, type):

    
    img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) 
    if type == 'candidate':
        thresholds = [0.6, 0.7, 0.8]
    else:
        thresholds = [0.6, 0.7, 0.8]

    bounding_boxes, landmarks = detect_faces(img, thresholds=thresholds)

    if len(bounding_boxes) > 0:
        width, height = img.size
        img_x_center = width/2
        img_y_center = height/2
        # print(width, height, img_center)

        y_box = []
        for i in range(len(bounding_boxes)):
            y_l_up = bounding_boxes[i][1]
            y_r_bottom = bounding_boxes[i][3]
            y_center = (y_l_up+y_r_bottom)/2

            if y_center<img_y_center:
                y_box.append(bounding_boxes[i])
        
        bounding_boxes = y_box
        new_bounding_boxes = np.array(bounding_boxes)
        new_landmarks = np.array(landmarks)

        box_center_dis = []
        for i in range(len(bounding_boxes)):
            y_l_up = bounding_boxes[i][1]
            y_r_bottom = bounding_boxes[i][3]
            y_center = (y_l_up+y_r_bottom)/2
            
            if y_center<img_y_center:
                x_l_up = bounding_boxes[i][0]
                x_r_bottom = bounding_boxes[i][2]
                x_center = (x_l_up+x_r_bottom)/2
                box_center_dis.append(abs(img_x_center-x_center))

        if len(box_center_dis)>0:
            min_dis = min(box_center_dis)
            min_dis_idx = box_center_dis.index(min_dis)
            # print(box_center_dis, min_dis_idx)

            bounding_boxes[min_dis_idx][0] = max(bounding_boxes[min_dis_idx][0]-20, 0)
            bounding_boxes[min_dis_idx][1] = max(bounding_boxes[min_dis_idx][1]-20, 0)
            bounding_boxes[min_dis_idx][2] = min(bounding_boxes[min_dis_idx][2]+20, width)
            bounding_boxes[min_dis_idx][3] = min(bounding_boxes[min_dis_idx][3]+20, height)

            new_bounding_boxes = np.array([bounding_boxes[min_dis_idx]])
            # print(new_bounding_boxes)
            new_landmarks = np.array([landmarks[min_dis_idx]])
        else:
            new_bounding_boxes = np.array([])
            new_landmarks = np.array([])
    
    else:
        new_bounding_boxes = bounding_boxes
        new_landmarks = landmarks

    # show_bboxes(img, bounding_boxes, landmarks)
    # vis_img = show_bboxes(img, new_bounding_boxes, new_landmarks)

    # vis_img.show()
    if len(new_bounding_boxes)>0:
        img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
        x1 = int(new_bounding_boxes[0][0])
        y1 = int(new_bounding_boxes[0][1])
        x2 = int(new_bounding_boxes[0][2])
        y2 = int(new_bounding_boxes[0][3])
        img = img[y1:y2, x1:x2,:]  
        # if type == 'candidate':
        #     cv2.imshow("x", img)
        #     cv2.waitKey(0)

        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        img = transform(img)

        return img
        
    elif type == 'cast':
        img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = transform(img)
        return img
    else:
        return None


# detect(cast_path, all_img_path, output_path, "cast")    
# detect(candidates_path, all_img_path, output_path, "candidate")    


    
    

    

