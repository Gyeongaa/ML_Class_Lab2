# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings(action = 'ignore') 
import argparse
import openslide
import cv2
import json
from segmentation_run import seg_model_load
import glob
# import pandas as pd
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from datetime import datetime
import pandas as pd


def save_result(img_input,save_path,patient_name,y_npy,cate):
    """ 
    병리이미지별 Segmentation mask 저장 (json)
    args:
        img_input(numpy): padding 및 resize 완료된 병리이미지
        save_path(str): 결과(json) 저장 경로
        patient_name(str): 결과(json) 파일명
        y_npy(numpy): 입력 Mask
        cate(str): Ground Truth 및 AI result 구분 파일명 (predict / true)

    """
    y_npy[y_npy==1]=255
    if not len(y_npy.shape)==2:
        y_npy=y_npy[0,:,:,0]
    y_npy=y_npy.astype("uint8")

    contours, hierachy=cv2.findContours(y_npy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_annotation=[]
    for c in range(len(contours)):
        contour_tmp=np.squeeze(contours[c],axis=1)
        contour_tmp=contour_tmp.reshape(-1)
        contour_tmp=contour_tmp.astype('float')
        contour_tmp=list(contour_tmp)
        roi_annotation_tmp={
                    "roi_number":c,
                    "mask":contour_tmp
                    }
        roi_annotation.append(roi_annotation_tmp)
    
    annotation_result={
                    "patient_name":patient_name,
                    "annotation":roi_annotation}

    os.makedirs("%s/indiv_json"%save_path,exist_ok=True)
    with open("%s/indiv_json/%s_%s.json"%(save_path,patient_name,cate), 'w', encoding='UTF8') as f:
        json.dump(annotation_result, f, indent=2, ensure_ascii=False)


def parse_args():
    # jpg_path: test하고자 하는 파일 경로
    # save_path: test 결과 저장 경로
    # output_path: 모델 예측 결과 저장 경로
    parser = argparse.ArgumentParser()
    parser.add_argument('--jpg_path',type=str,default="../test")
    parser.add_argument('--save_path',type=str,default="../test/result")
    parser.add_argument('--output_path', type=str, default='../test/mask_pred')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    save_path=args.save_path
    output_path=args.output_path
    os.makedirs(output_path, exist_ok=True)

    # 학습된 모델 로드
    run_model=seg_model_load()

    # 테스트 할 이미지 목록
    test_list=glob.glob("%s/image/*.jpg"%args.jpg_path)

    # Execution
    for test_count, test_path in enumerate(test_list):
        test_path=test_path.replace("\\","/")
        patient_name=test_path.split("/")[-1].split(".jpg")[0]

        ### inference image Load
        img_input=cv2.imread(test_path)

        ### model inference
        segmentation_result=run_model.inference_run(img_input)

        # Save predicted image
        y_npy = segmentation_result.copy()
        y_npy[y_npy==1]=255
        if not len(y_npy.shape)==2:
            y_npy=y_npy[0,:,:,0]

        # 최종 타입 변경 
        y_npy=y_npy.astype("uint8")

        cv2.imwrite(output_path + '/'  + patient_name + '.jpg', y_npy)

        print(f'{patient_name} mask is created!')

if __name__ == '__main__':
    main()


          
