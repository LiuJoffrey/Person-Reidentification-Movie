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
        # thresholds = [0.3, 0.4, 0.5]
    else:
        thresholds = [0.6, 0.7, 0.8]
    if type == 'cast':
        count=0
        bounding_boxes = []
        while len(bounding_boxes)==0:
            thresholds = [0.6-count, 0.7-count, 0.8-count]
            bounding_boxes, landmarks = detect_faces(img, thresholds=thresholds)
            count+=1
            # print(count)
            if count == 5:
                break
            # if len(bounding_boxes)>0:
            #     vis_img = show_bboxes(img, bounding_boxes, landmarks)
            #     vis_img.show()
        # exit()

    else:    
        bounding_boxes, landmarks = detect_faces(img, thresholds=thresholds)

    if len(bounding_boxes) > 0 and type == 'cast':
        width, height = img.size
        img_x_center = width/2
        img_y_center = height/2
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

            bounding_boxes = np.array([bounding_boxes[min_dis_idx]])
            # print(new_bounding_boxes)
            landmarks = np.array([landmarks[min_dis_idx]])

    


    if len(bounding_boxes) > 0 and type == 'candidate':
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
        box_size = []
        for i in range(len(bounding_boxes)):
            y_l_up = bounding_boxes[i][1]
            y_r_bottom = bounding_boxes[i][3]
            y_len = abs(y_r_bottom-y_l_up)
            x_l_up = bounding_boxes[i][0]
            x_r_bottom = bounding_boxes[i][2]
            x_len = abs(x_r_bottom-x_l_up)
            bound_size = x_len*y_len
            box_size.append(bound_size)

            # y_center = (y_l_up+y_r_bottom)/2

            # if y_center<img_y_center:
            #     x_l_up = bounding_boxes[i][0]
            #     x_r_bottom = bounding_boxes[i][2]
            #     x_center = (x_l_up+x_r_bottom)/2
            #     box_center_dis.append(abs(img_x_center-x_center))

        if len(box_size)>0:
            max_size = max(box_size)
            max_size_idx = box_size.index(max_size)
            bounding_boxes[max_size_idx][0] = max(bounding_boxes[max_size_idx][0]-20, 0)
            bounding_boxes[max_size_idx][1] = max(bounding_boxes[max_size_idx][1]-20, 0)
            bounding_boxes[max_size_idx][2] = min(bounding_boxes[max_size_idx][2]+20, width)
            bounding_boxes[max_size_idx][3] = min(bounding_boxes[max_size_idx][3]+20, height)

            # min_dis = min(box_center_dis)
            # min_dis_idx = box_center_dis.index(min_dis)
            # # print(box_center_dis, min_dis_idx)

            # bounding_boxes[min_dis_idx][0] = max(bounding_boxes[min_dis_idx][0]-20, 0)
            # bounding_boxes[min_dis_idx][1] = max(bounding_boxes[min_dis_idx][1]-20, 0)
            # bounding_boxes[min_dis_idx][2] = min(bounding_boxes[min_dis_idx][2]+20, width)
            # bounding_boxes[min_dis_idx][3] = min(bounding_boxes[min_dis_idx][3]+20, height)

            # new_bounding_boxes = np.array([bounding_boxes[min_dis_idx]])
            # # print(new_bounding_boxes)
            # new_landmarks = np.array([landmarks[min_dis_idx]])

            new_bounding_boxes = np.array([bounding_boxes[max_size_idx]])
            # print(new_bounding_boxes)
            new_landmarks = np.array([landmarks[max_size_idx]])
        else:
            new_bounding_boxes = np.array([])
            new_landmarks = np.array([])
    
    else:
        
        new_bounding_boxes = bounding_boxes
        new_landmarks = landmarks
    
    # show_bboxes(img, bounding_boxes, landmarks)
    # vis_img = show_bboxes(img, new_bounding_boxes, new_landmarks)
    # vis_img = cv2.cvtColor(np.asarray(vis_img),cv2.COLOR_RGB2BGR)
    # # vis_img.show()
    # print(new_bounding_boxes)
    # cv2.imshow("x", vis_img)
    # cv2.waitKey(0)
    # return 
    
    if len(new_bounding_boxes)>0:

        img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)

        # print(img.shape)
        # print(new_bounding_boxes)
        # exit()
        
        x1 = int(new_bounding_boxes[0][0])
        y1 = int(new_bounding_boxes[0][1])
        x2 = int(new_bounding_boxes[0][2])
        y2 = int(new_bounding_boxes[0][3])
        
        
        img = img[y1:y2, x1:x2,:]  
        # print(img)
        # print(img.shape)
        # print(new_bounding_boxes)
        # img.show()
        # cv2.imshow("x", img)
        # cv2.waitKey(0)

        
        # print(img.shape)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        # exit()
        # if type == 'candidate':
        #     cv2.imshow("x", img)
        #     cv2.waitKey(0)

        # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

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


    
    

    

