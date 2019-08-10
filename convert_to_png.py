#! /usr/bin/python3
import os
import cv2

data_set = './Data/new_data/merged'
for i,image_name in enumerate(os.listdir(data_set)):
    if image_name[-3:] == 'jpg':
        image_path = os.path.join(data_set,image_name)
        decoded = cv2.imread(image_path)
        new_path = os.path.join(data_set, image_name[:-3]+'png')
        cv2.imwrite(new_path, decoded)
        print(i, end='\r')