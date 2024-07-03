import numpy as np
import cv2
import json
import matplotlib.path as mpltPath
import os
import warnings
from math import sqrt
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import datetime
import time


def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

def make_csv(tile_invert_path, csv_save_path, label, label_name):
    path = []
    
    for dira in glob(tile_invert_path):
        path.append(dira)

    raw_data = {'path' : path, 'class' : label}

    data = pd.DataFrame(raw_data)
    data.to_csv(csv_save_path + label_name + '.csv', index=False)
    
def combine_csv(csv_save_path):
    # 디렉토리 내의 모든 CSV 파일 목록 가져오기
    csv_files = [f for f in os.listdir(csv_save_path) if f.endswith('.csv')]

    # 빈 DataFrame을 생성하고 CSV 파일들을 순회하며 병합
    merged_data = pd.DataFrame()
    for csv_file in csv_files:
        file_path = os.path.join(csv_save_path, csv_file)
        data = pd.read_csv(file_path)
        merged_data = pd.concat([merged_data, data], ignore_index=True)

    # 병합된 데이터를 새로운 CSV 파일로 저장
    merged_data.to_csv('./data/csv/test.csv', index=False)
    
def main(normal_tile_invert_path, insitu_tile_invert_path, malignant_tile_invert_path, csv_save_path):
    ###### Test csv 시작 ######
    start = time.time()
    d = datetime.datetime.now()
    now_time = f"{d.year}-{d.month}-{d.day} {d.hour}:{d.minute}:{d.second}"
    print(f'[Test csv Start]')
    print(f'Test csv Start Time : {now_time}')

    createDirectory(csv_save_path)

    make_csv(normal_tile_invert_path, csv_save_path, '0', '0_normal')
    make_csv(insitu_tile_invert_path, csv_save_path, '1', '1_insitu')
    make_csv(malignant_tile_invert_path, csv_save_path, '2', '2_malignant')

    combine_csv(csv_save_path)

    end = time.time()
    d = datetime.datetime.now()
    now_time = f"{d.year}-{d.month}-{d.day} {d.hour}:{d.minute}:{d.second}"
    print(f'Test csv Time : {now_time}s Time taken : {end-start}')
    print(f'[Test csv End]')
    ###### Model prediction 끝 ######