#!/bin/bash
# 데이터 전처리 
python training_datacreate.py
python training_dataset_split.py

# 모델 학습 
python segmentation_train.py