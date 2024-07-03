# -*- coding: utf-8 -*-

import os
import warnings
warnings.filterwarnings(action = 'ignore') 
os.environ["SM_FRAMEWORK"] = "tf.keras" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from keras.models import load_model
import tensorflow as tf
from functional import jaccard_loss,iou_score

class seg_model_load():

    ### Segmentation Model Load
    model=load_model("../model/segmentation_model",custom_objects={'jaccard_loss':jaccard_loss,'iou_score':iou_score})
    
    def __init__(self):

        ### AI Model 결과 Threshold 값 설정
        self.prob=0.5


    def inference_run(self, image):

        ### Input Image 픽셀 단위 전처리
        image = image.astype(np.float32)
        image /= 255.
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image -= mean
        image /= std

        ### AI model inference Run
        result=seg_model_load.model.predict(np.expand_dims(image,axis=0),verbose=0)

        ### Threshold 값 기준 이진화
        result[result<self.prob]=0
        result[result>=self.prob]=1
        return result

