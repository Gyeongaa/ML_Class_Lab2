# Import library 
import os
import random 
import shutil
import time
import numpy as np

#######################################################################
source_folder = '../output/'
mask_folder = '../output/mask/'
target_folder = '../output/dataset/'
#######################################################################

# create target folder 
os.makedirs(target_folder, exist_ok=True)

# 인공지능 학습용 폴더 별 데이터 셋 구성
def CreateDataset(class_folder, source_folder, target_folder):
    """
    전체 데이터 셋 Train, validation, test 셋 분할하는 로직 

    Input: soruce_folder, target_folder
    Output: 각 클래스 별 비율이 적용된 Split 데이터 셋 폴더
    """

    # 클래스 폴더 내 이미지 파일 리스트 가져오기 
    image_files = os.listdir(source_folder)

    # 이미지 파일을 섞어서 순서를 무작위로 만듬
    random.seed(42) # 고정
    random.shuffle(image_files)

    # 이미지 비율 계산
    image_files = os.listdir(source_folder)
    total_images = len(image_files)
    train_ratio = 0.8
    valid_ratio = 0.2

    train_count = int(train_ratio * total_images)

    # 분할된 이미지를 대상 폴더로 이동 
    for i, image_file in enumerate(image_files):
        source_image_path = os.path.join(source_folder, image_file)
        if i < train_count:
            target_split_folder = 'train'
        else:
            target_split_folder = 'val'


        # Target 폴더 설정
        target_image_folder = os.path.join(target_folder, target_split_folder, class_folder, class_folder)  
        target_image_path = os.path.join(target_image_folder, image_file)

        # 대상 폴더가 없을 경우 생성 
        os.makedirs(target_image_folder, exist_ok=True)

        # 이미지 파일 이동 
        shutil.copy(source_image_path, target_image_path)

# 분할 진행

class_folders = ['image','mask']

for class_folder in class_folders:
    result = CreateDataset(class_folder, source_folder + class_folder, target_folder )

print('이미지 분할이 완료되었습니다.')
    