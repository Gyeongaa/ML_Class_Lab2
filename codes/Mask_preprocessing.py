import os
import numpy as np 
from glob import glob
from PIL import Image
import cv2
import slideio
import json
from skimage.draw import polygon2mask
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import warnings
import datetime
warnings.filterwarnings("ignore")

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

def size_ratio(scene,img_size):
    width = scene.rect[2]
    height = scene.rect[3]
    ratio=0
    inverse_ratio=0
    img_width=0
    img_height=0
    if width>height:
        ratio=img_size/width
        inverse_ratio=width/img_size
        img_width=img_size
        img_height=height*ratio
    else:
        ratio=img_size/height
        inverse_ratio=height/img_size
        img_height=img_size
        img_width=width*ratio
        
    return int(img_width),int(img_height),inverse_ratio


def normal_make_mask(whi_list, json_list, mask_save_path):
    for i in tqdm(range(len(whi_list))):
        slide = slideio.open_slide(whi_list[i], "GDAL")
        fileName=os.path.basename(os.path.splitext(whi_list[i])[0])
        num_scenes = slide.num_scenes
        scene = slide.get_scene(0)
        img_width,img_height,ratio=size_ratio(scene,2048)
        svsWidth = scene.rect[2]
        svsHeight = scene.rect[3]
        slide_block = scene.read_block((0, 0, svsWidth, svsHeight),size=(int(img_width), int(img_height)))
        image=cv2.cvtColor(slide_block, cv2.COLOR_BGR2RGB)
        dst_mask=np.zeros((img_height,img_width),dtype=np.uint8)
        with open(json_list[i]) as f:
            json_object = json.load(f)
        polygon_count=len(json_object['files'][0]['objects'])
        image_shape=(img_height,img_width)
        for j in range(polygon_count):
            if json_object['files'][0]['objects'][j]['label']=='NT_normal':
                polygon=np.array(json_object['files'][0]['objects'][j]['coordinate'])*1/ratio
                polygon1=np.copy(polygon)
                polygon1[:,0]=polygon[:,1]
                polygon1[:,1]=polygon[:,0]
                mask=polygon2mask(image_shape,polygon1)
                dst_mask=mask+dst_mask
        dst_mask=np.where(dst_mask>0,255,0)
        dst_mask=cv2.cvtColor(dst_mask.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        cv2.imwrite(mask_save_path+fileName+'.tiff', dst_mask )


def tumor_make_mask(whi_list, json_list, mask_save_path):
    for i in tqdm(range(len(whi_list))):
        slide = slideio.open_slide(whi_list[i], "GDAL")
        fileName=os.path.basename(os.path.splitext(whi_list[i])[0])
        num_scenes = slide.num_scenes
        scene = slide.get_scene(0)
        img_width,img_height,ratio=size_ratio(scene,2048)
        svsWidth = scene.rect[2]
        svsHeight = scene.rect[3]
        slide_block = scene.read_block((0, 0, svsWidth, svsHeight),size=(int(img_width), int(img_height)))
        image=cv2.cvtColor(slide_block, cv2.COLOR_BGR2RGB)
        dst_mask=np.zeros((img_height,img_width),dtype=np.uint8)
        with open(json_list[i]) as f:
            json_object = json.load(f)
        polygon_count=len(json_object['files'][0]['objects'])
        image_shape=(img_height,img_width)
        for j in range(polygon_count):
            if json_object['files'][0]['objects'][j]['label']=='TP_tumor':
                polygon=np.array(json_object['files'][0]['objects'][j]['coordinate'])*1/ratio
                polygon1=np.copy(polygon)
                polygon1[:,0]=polygon[:,1]
                polygon1[:,1]=polygon[:,0]
                mask=polygon2mask(image_shape,polygon1)
                dst_mask=mask+dst_mask
        dst_mask=np.where(dst_mask>0,255,0)
        dst_mask=cv2.cvtColor(dst_mask.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        cv2.imwrite(mask_save_path+fileName+'.tiff', dst_mask )
        
        
def main(normal_whi_list, insitu_whi_list, malignant_whi_list, normal_mask_path, insitu_mask_path, malignant_mask_path, normal_json_list, insitu_json_list, malignant_json_list):
    ###### Mask 만들기 시작 ######
    start = time.time()
    d = datetime.datetime.now()
    now_time = f"{d.year}-{d.month}-{d.day} {d.hour}:{d.minute}:{d.second}"
    print(f'[Make mask Start]')
    print(f'Make mask Start Time : {now_time}')

    createDirectory(normal_mask_path)
    createDirectory(insitu_mask_path)
    createDirectory(malignant_mask_path)

    normal_make_mask(normal_whi_list, normal_json_list, normal_mask_path)
    tumor_make_mask(insitu_whi_list, insitu_json_list, insitu_mask_path)
    tumor_make_mask(malignant_whi_list, malignant_json_list, malignant_mask_path)

    end = time.time()
    d = datetime.datetime.now()
    now_time = f"{d.year}-{d.month}-{d.day} {d.hour}:{d.minute}:{d.second}"
    print(f'Make mask Time : {now_time}s Time taken : {end-start}')
    print(f'[Mask mask End]')
    ###### Mask 만들기 끝 ######