# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings(action = 'ignore') 
import argparse
import openslide
import cv2
import json
import glob
import os
import numpy as np
from datetime import datetime


def data_load(data_path):
    """ 
    tiff 데이터 Load 
    args:
        data_path(str): tiff 데이터 경로
    
    return:
        image(numpy): 다운샘플링된 병리이미지
        slide(dict): tiff meta data

    """

    print("data_load...")
    scale=8.0
    print(data_path)
    slide = openslide.open_slide(data_path)
    possible_level = slide.level_downsamples
    possible_elements = [int(i - scale) for i in possible_level]
    level = possible_elements.index(0)
    new_dimension = slide.level_dimensions[level]
    image=np.asarray(slide.get_thumbnail(new_dimension))
    return image,slide

def otsu_preprocess(img):
    """ 
    Otsu Thresholding 값을 활용한 이미지 Crop
    args:
        img(numpy): tiff에서 추출된 다운샘플링 이미지
    
    return:
        img(numpy): Otsu Threshoding min,max 값을 활용한 crop된 image
        [otsu_x_min,otsu_x_max,otsu_y_min,otsu_y_max](list): image crop에 사용된 좌표값

    """
    print("otsu_preprocess...")
    img_lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    t, t_otsu = cv2.threshold(img_lab[:,:,1], 0, 255,  cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    contours, h = cv2.findContours(t_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    mask=np.zeros((img.shape),dtype='uint8')
    for c in range(len(contours)):
        if len(contours[c]) > 100:
            mask=cv2.fillConvexPoly(mask,np.int32(contours[c]),(255,255,255))

    xy_list=np.where(mask==255)
    otsu_x_min,otsu_x_max,otsu_y_min,otsu_y_max=xy_list[0].min(),xy_list[0].max(),xy_list[1].min(),xy_list[1].max()
    img=img[otsu_x_min:otsu_x_max, otsu_y_min:otsu_y_max, :]
    return img, [otsu_x_min,otsu_x_max,otsu_y_min,otsu_y_max]

def model_preprocess(image,i_m):
    """ 
    이미지 Padding 및 Resize
    args:
        image(numpy): Otsu Threshoding min,max 값을 활용한 crop된 image
        i_m(str): image, mask 구분
    
    return:
        image(numpy): 이미지 Padding 및 1024,1024로 Resize된 image

    """
    print("model_preprocess...")
    image_re_size=1024
    image_height, image_width = image.shape[:2]
    if image_height > image_width:
        scale = image_re_size / image_height
        resized_height = image_re_size
        resized_width = int(image_width * scale)
    else:
        scale = image_re_size / image_width
        resized_height = int(image_height * scale)
        resized_width = image_re_size
    image = cv2.resize(image, (resized_width, resized_height))
    pad_h = image_re_size - resized_height
    pad_w = image_re_size - resized_width
    if i_m == 'image':
        image = np.pad(image, [(0, pad_h), (0, pad_w), (0, 0)], mode='constant')
    elif i_m == 'mask':
        image = np.pad(image, [(0, pad_h), (0, pad_w)], mode='constant')
    else:
        print("model_preprocess error", i_m)
    return image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tiff_path',type=str,default="../test/wsi")
    parser.add_argument('--save_path',type=str,default="../test")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    tiff_path=args.tiff_path
    save_path=args.save_path

    # 폴더 생성
    os.makedirs(save_path+"/mask", exist_ok=True)
    os.makedirs(save_path+"/image", exist_ok=True)

    #### preprocess 시작 
    test_list =glob.glob(tiff_path + '/*')
    for file in test_list:
        patient_name = file.split('/')[-1][:-5]

        #### tiff에서 다운샘플링된 이미지 Load
        wsi_raw_img, slide = data_load(file)

        #### Otsu Thresholding 값을 활용한 이미지 Crop
        wsi_otsu_img = otsu_preprocess(wsi_raw_img)

        #### 이미지 Padding 및 Resize
        wsi_otsu_img = model_preprocess(wsi_otsu_img[0],'image')

        ### 전처리 완료된 이미지 및 마스크 저장
        cv2.imwrite(save_path+"/image/%s.jpg"%patient_name,wsi_otsu_img)

        print(f'{file[:-5]} is created!')


if __name__ == '__main__':
    main()