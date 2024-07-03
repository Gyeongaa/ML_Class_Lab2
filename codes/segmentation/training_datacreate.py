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

def mask_preprocess(image_path,otsu_crop_coord,raw_image_shape,slide):
    """ 
    Ground Truth Mask 전처리 (crop, 이진화, Padding, Resize)
    args:
        image_path(str): Ground Truth Mask 파일 경로
        otsu_crop_coord(list): image crop에 사용된 좌표값
        raw_image_shape(list): Raw image 사이즈
        slide(dict): tiff meta 정보
    
    return:
        mask(numpy): 이미지 crop, 이진화, Padding, Resize된 mask image

    """

    with open(image_path.replace("wsi","annotation").replace(".tiff",".json")) as json_file:
        data = json.load(json_file)
    obj = data['files'][0]['objects']
    mask=np.zeros((slide.dimensions[1],slide.dimensions[0]),dtype='uint8')
    for i in range(len(obj)):
        label = obj[i]['label']
        if label in ['TP_benign','TP_malignant','NT_normal_renal', 'NT_normal_extrarenal']:
            coordinate = obj[i]['coordinate']
            paths_xy=np.array([list(map(int,list(t))) for t in coordinate])
            mask=cv2.fillPoly(mask,np.int32([paths_xy]),(255,255,255))
            
    mask=cv2.resize(mask, dsize=(raw_image_shape[1],raw_image_shape[0]))
    mask[mask>=128]=255
    mask[mask<128]=0
    mask=mask[otsu_crop_coord[0]:otsu_crop_coord[1], otsu_crop_coord[2]:otsu_crop_coord[3]]
    mask=model_preprocess(mask,"mask")
    mask[mask>=128]=255
    mask[mask<128]=0
    return mask

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tiff_path',type=str,default="../data/wsi")
    parser.add_argument('--save_path',type=str,default="../output")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    tiff_path=args.tiff_path
    save_path=args.save_path

    ### 저장 폴더 생성
    os.makedirs(save_path +'/mask', exist_ok=True)
    os.makedirs(save_path +'/image', exist_ok=True)
    os.makedirs(save_path +'/result', exist_ok=True)

    # Log file
    with open("%s/result/log.txt"%save_path, "w") as f_log:

        #### code 실행시 적용된 args값 log 기록
        f_log.write("sh segmentation_inference.sh\n")
        args_key_list=vars(args).keys()
        arg_txt=""
        for args_key in args_key_list:
            arg_txt = arg_txt+" --%s %s"%(args_key,vars(args)[args_key])
        txt="python %s%s"%(__file__.split("/")[-1],arg_txt)
        f_log.write(txt+"\n")

        try:
            start_time = datetime.now()
            test_list =glob.glob(tiff_path + '/*')

            #### preprocess 시작 시간 log 기록
            f_log.write("preprocessing start time: "+str(start_time)+"\n")

            # Execution 
            for file in test_list:
                patient_name = file.split('/')[-1][:-5]

                #### tiff에서 다운샘플링된 이미지 Load
                wsi_raw_img,slide=data_load(file)
                raw_image_shape=wsi_raw_img.shape

                #### Otsu Thresholding 값을 활용한 이미지 Crop
                wsi_otsu_img,otsu_crop_coord=otsu_preprocess(wsi_raw_img[:,:,::-1])

                #### 이미지 Padding 및 Resize
                wsi_otsu_img=model_preprocess(wsi_otsu_img,'image')

                #### Ground Truth Mask 전처리 (crop, 이진화, Padding, Resize)
                mask_img=mask_preprocess(file,otsu_crop_coord,raw_image_shape,slide)

                ### 전처리 완료된 이미지 및 마스크 저장
                cv2.imwrite(save_path+"/image/%s.jpg"%patient_name,wsi_otsu_img)
                cv2.imwrite(save_path+"/mask/%s.jpg"%patient_name,mask_img)

            end_time = datetime.now()
            f_log.write("preprocessing end time: "+str(end_time)+"\n")
            f_log.write("\n")

        except Exception as e:
            f_log.write(str(e)+"\n")
            f_log.write("\n")


if __name__ == '__main__':
    main()


          
