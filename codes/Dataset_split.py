# Import library 
import os
import random 
import shutil
import time

# 코드 시작 타임 
start_time = time.time()

###########################################################
# 기본 경로 설정
source_folder = '../output/tiles'
target_folder = '../output/tiles_dataset'
###########################################################

# 폴더 리스트 
folders = os.listdir(source_folder)

# Target folder 생성 
os.makedirs(target_folder, exist_ok=True)

# 인공지능 학습용 폴더 별 데이터 셋 구성
def CreateDataset(folders, source_folder, target_folder):
    """
    전체 데이터 셋 Train, validation, test 셋 분할하는 로직 

    Input: soruce_folder, target_folder
    Output: 각 클래스 별 비율이 적용된 Split 데이터 셋 폴더
    """
    for class_folder in folders:
        class_path = os.path.join(source_folder, class_folder)

        # 클래스 폴더 내 이미지 파일 리스트 가져오기 
        image_files = os.listdir(class_path)

        # 이미지 파일을 섞어서 순서를 무작위로 만듬 
        random.shuffle(image_files)

        # 이미지 비율 계산
        total_images = len(image_files)
        train_ratio = 0.7
        valid_ratio = 0.2

        train_count = int(train_ratio * total_images)
        valid_count = int(valid_ratio * total_images)

        # 분할된 이미지를 대상 폴더로 이동 
        for i, image_file in enumerate(image_files):
            source_image_path = os.path.join(class_path, image_file)
            if i < train_count:
                target_split_folder = 'train'
            elif i < train_count + valid_count:
                target_split_folder = 'valid'
            else:
                target_split_folder = 'test'
            
            # Target 폴더 설정
            target_image_folder = os.path.join(target_folder, target_split_folder, class_folder)
            target_image_path = os.path.join(target_image_folder, image_file)

            # 대상 폴더가 없을 경우 생성 
            os.makedirs(target_image_folder, exist_ok=True)

            # 이미지 파일 이동 
            shutil.move(source_image_path, target_image_path)
        
    print('이미지 분할이 완료되었습니다.')


# Execution
CreateDataset(folders, source_folder, target_folder)

# Time 
end_time = time.time()
execution_time = end_time - start_time 

print(f"코드 실행 시간: {execution_time}초")
