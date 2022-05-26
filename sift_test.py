import numpy as np
from models.custom_components import HessianKernelGood
import cv2 as cv
import torch
from tqdm import tqdm
import os
import json


# image = cv.imread('/Users/justinbutler/Desktop/school/Calgary/ML_Work/Datasets/Shapes/Triangles_640_7500imgs_mod4/images/image_16.png')
# gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
# sift = cv.SIFT_create()
# kp, des = sift.detectAndCompute(gray ,None)
# img = cv.drawKeypoints(gray, kp, image)
# cv.imwrite('sift_keypoints.jpg', img)

# image = np.transpose(image, (2,0,1))
# tensor = torch.from_numpy(image)
# hessian = HessianKernelGood(scale=0)
# keys = hessian(tensor.float().unsqueeze(dim=0))
# print(keys)

img_dir = '/home/justin.butler1/Data/uav-detect/cars-only/dataset3/images'
new_json_file = '/home/justin.butler1/Data/uav-detect/cars-only/dataset3/dataset3_cars_keys_sift.json'
old_json = '/home/justin.butler1/Data/uav-detect/cars-only/dataset3/dataset3_x1y1wh.json'
key_list = []
json_list = []

img_list = os.listdir(img_dir)

with open(old_json, 'r') as j:
    jdata = json.load(j)

img_info = jdata['images']
key_store = {}

for img in tqdm(img_list, desc='Images Done: '):
    key_store = {}
    if img[-3:] != 'jpg' and img[-3:] != 'png':
        continue

    key_list = []
    file_name = img[2].split('/')[-1]
    file_name = img
    image = cv.imread(f'{img_dir}/{img}')
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    sift = cv.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)

    if len(kp) == 0:
        keys = [list(np.zeros(132))]
        key_list.append(keys)
    else:
        for key, descrip in zip(kp, des):
            x, y = key.pt
            scale = key.size
            angle = key.angle
            descrip_flt = [float(item) for item in list(descrip)]
            full_tens = [x, y, angle, scale] + descrip_flt
            key_list.append(full_tens)

    for img in img_info:
        # if img['file_name'].split('/')[-1] == file_name:
        if img['file_name'] == file_name:
            img_id = img['id']
            break

    key_store['image_id'] = img_id  # file_name[:-4]

    # if len(keys) == 0:
    #     keys = [list(np.zeros(132))]

    key_store['keys'] = key_list
    json_list.append(key_store)

print(os.getcwd())
with open(new_json_file, 'w') as j:
    json.dump(json_list, j)
