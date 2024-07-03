import os
import numpy as np 
from glob import glob
import cv2
import slideio
import json
from skimage.draw import polygon2mask
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import warnings
import datetime
import math
from PIL import Image, ImageOps
warnings.filterwarnings("ignore")

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")


def make_tile(whi_list, mask_file, tile_save_path):
    slide_tile_size=2048
    maske_files=mask_file
    folder_list=[f[:-5] for f in maske_files]
    for i in tqdm(range(len(whi_list))):
        count=0
        fileName=os.path.basename(os.path.splitext(maske_files[i])[0])
        slide = slideio.open_slide(whi_list[i], "GDAL")
        scene = slide.get_scene(0)
        svsWidth = scene.rect[2]
        svsHeight = scene.rect[3]
        mask_file=maske_files[i]
        folder=[s for s in folder_list if fileName in s][0]
        mask_image=np.array(Image.open(mask_file))
        ratio=0
        if svsWidth>svsHeight:
            ratio=svsWidth/2048
        else:
            ratio=svsHeight/2048
        inverse_ratio=math.floor(1/ratio*10000)/10000
        for widthCount in range(0, int(svsWidth // slide_tile_size)):
                for heightCount in range(0, int(svsHeight // slide_tile_size)):
                    point_x =np.linspace(widthCount*slide_tile_size,widthCount*slide_tile_size+slide_tile_size-1,slide_tile_size,dtype=np.int32)
                    point_y =np.linspace(heightCount*slide_tile_size,heightCount*slide_tile_size+slide_tile_size-1,slide_tile_size,dtype=np.int32)
                    point=np.meshgrid(point_x,point_y)
                    mask_point=np.copy(point)
                    mask_point[0]=(mask_point[0]*inverse_ratio).astype(np.int64)
                    mask_point[1]=(mask_point[1]*inverse_ratio).astype(np.int64)
                    if mask_point[0].max()==mask_image.shape[1]:
                        mask_point[0]-=1
                    if mask_point[1].max()==mask_image.shape[0]:
                        mask_point[1]-=1
                    try:
                        tile_mask_image=mask_image[mask_point[1],mask_point[0]]/255
                        if tile_mask_image.max()>0:
                            count+=1
                            image = scene.read_block((widthCount * slide_tile_size, heightCount * slide_tile_size, slide_tile_size, slide_tile_size),size=(512,512))
                            img=Image.fromarray(image)
                            img.save(tile_save_path+'/'+fileName+'_'+str(count)+'.jpg')
                        
                    except:
                        file = fileName+'_'+str(count) 
                        print(fileName+'_'+str(count))
                        
                        
def invert(tile_image_list, save_image_list):
    for i in tqdm(range(len(tile_image_list))):
        img=Image.open(tile_image_list[i])
        img= ImageOps.invert(img)
        img.save(save_image_list[i])
        
        
def main1(normal_whi_list, insitu_whi_list, malignant_whi_list, normal_mask_files, insitu_mask_files, malignant_mask_files, normal_tile_path, insitu_tile_path, malignant_tile_path):
    ###### Tile 만들기 시작 ######
    start = time.time()
    d = datetime.datetime.now()
    now_time = f"{d.year}-{d.month}-{d.day} {d.hour}:{d.minute}:{d.second}"
    print(f'[Make tile Start]')
    print(f'Make tile Start Time : {now_time}')

    createDirectory(normal_tile_path)
    createDirectory(insitu_tile_path)
    createDirectory(malignant_tile_path)

    make_tile(normal_whi_list, normal_mask_files, normal_tile_path)
    make_tile(insitu_whi_list, insitu_mask_files, insitu_tile_path)
    make_tile(malignant_whi_list, malignant_mask_files, malignant_tile_path)

    end = time.time()
    d = datetime.datetime.now()
    now_time = f"{d.year}-{d.month}-{d.day} {d.hour}:{d.minute}:{d.second}"
    print(f'Make tile Time : {now_time}s Time taken : {end-start}')
    print(f'[Mask tile End]')
    ###### Tile 만들기 끝 ######


def main2(normal_tile_path, insitu_tile_path, malignant_tile_path, normal_tile_invert_path, insitu_tile_invert_path, malignant_tile_invert_path):
    ###### Tile invert 시작 ######
    start = time.time()
    d = datetime.datetime.now()
    now_time = f"{d.year}-{d.month}-{d.day} {d.hour}:{d.minute}:{d.second}"
    print(f'[Tile invert Start]')
    print(f'Tile invert Start Time : {now_time}')

    createDirectory(normal_tile_invert_path)
    createDirectory(insitu_tile_invert_path)
    createDirectory(malignant_tile_invert_path)

    invert(normal_tile_path, normal_tile_invert_path)
    invert(insitu_tile_path, insitu_tile_invert_path)
    invert(malignant_tile_path, malignant_tile_invert_path)

    end = time.time()
    d = datetime.datetime.now()
    now_time = f"{d.year}-{d.month}-{d.day} {d.hour}:{d.minute}:{d.second}"
    print(f'Tile invert Time : {now_time}s Time taken : {end-start}')
    print(f'[Tile invert End]')
    ###### Tile 만들기 끝 ######
