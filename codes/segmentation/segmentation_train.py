# -*- coding: utf-8 -*-
import os 
os.environ["SM_FRAMEWORK"] = "tf.keras" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from segmentation_models import Unet
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
import numpy as np
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import argparse

def preprocess_input(image):
  image = image.astype(np.float32)
  image /= 255.
  mean = [0.485, 0.456, 0.406]
  std = [0.229, 0.224, 0.225]
  image -= mean
  image /= std
  return image

def mask_process(x): 
  x[x<128]=0
  x[x>=128]=1 
  return x

def sch(epoch):
  if epoch>100:
    return 0.000001
  elif epoch>50 and epoch<=100:
    return 0.00001
  else:
    return 0.0001

def parse_args():
  ### train, val 데이터는 segmentation_datacreate.py 코드로 생성
  ### train_path: 학습 이미지 / 마스크 경로 ( train_path/image/이미지, train_path/mask/이미지 파일명과 동일한 마스크 이미지)
  ### val_path:   검증 이미지 / 마스크 경로 ( val_path/image/이미지, val_path/mask/이미지 파일명과 동일한 마스크 이미지)
  ### save_path:  모델 저장 경로 
  ### batch_size: 배치사이즈 설정
  ### image_size: 모델 input 이미지 사이즈 설정
  ### epoch:      epoch 설정
  parser = argparse.ArgumentParser()
  parser.add_argument('--train_path',type=str,default="../output/dataset/train")
  parser.add_argument('--val_path',type=str,default="../output/dataset/val")
  parser.add_argument('--save_path',type=str,default="../model")
  parser.add_argument('--batch_size',type=int,default=1)
  parser.add_argument('--image_size',type=int,default=1024)
  parser.add_argument('--epoch',type=int,default=10)
  args = parser.parse_args()
  return args

def main():
  args = parse_args()
  train_path=args.train_path
  val_path=args.val_path
  save_path=args.save_path
  batch_size=args.batch_size
  image_re_size=args.image_size
  epoch_number=args.epoch

  ### 모델 정의
  model = Unet('efficientnetb0', classes=1, encoder_weights='imagenet', decoder_use_batchnorm=True, decoder_filters=((1024, 512, 256, 128, 64)))
  model.compile(Adam(lr=1e-4), loss=bce_jaccard_loss, metrics=[iou_score])

  ### data augmentation
  print("model run")
  sc=LearningRateScheduler(sch)
  data_gen_args = dict(horizontal_flip=True,
                          rotation_range=50,
                          width_shift_range=0.6,
                          height_shift_range=0.6,
                          zoom_range=0.6,fill_mode='constant',cval=0,preprocessing_function=preprocess_input)


  data_gen_args_ = dict(horizontal_flip=True,
                    rotation_range=50,
                    width_shift_range=0.6,
                    height_shift_range=0.6,
                    zoom_range=0.6,fill_mode='constant',cval=0,preprocessing_function=mask_process)
  
  data_gen_args_val_img=dict(preprocessing_function=preprocess_input)
  data_gen_args_val_mask = dict(preprocessing_function=mask_process)

  image_datagen = ImageDataGenerator(**data_gen_args)
  mask_datagen = ImageDataGenerator(**data_gen_args_)
  image_val_datagen = ImageDataGenerator(**data_gen_args_val_img)
  mask_val_datagen = ImageDataGenerator(**data_gen_args_val_mask)

  seed = 12
  #'/image'
  image_generator = image_datagen.flow_from_directory(train_path,class_mode=None,seed=seed,target_size=(image_re_size, image_re_size),batch_size=batch_size)
  mask_generator = mask_datagen.flow_from_directory(train_path,class_mode=None,seed=seed,target_size=(image_re_size, image_re_size),batch_size=batch_size, color_mode="grayscale")
  image_validation_generator = image_val_datagen.flow_from_directory(val_path,class_mode=None,seed=seed,target_size=(image_re_size, image_re_size),batch_size=batch_size)
  mask_validation_generator = mask_val_datagen.flow_from_directory(val_path,class_mode=None,seed=seed,target_size=(image_re_size, image_re_size),batch_size=batch_size, color_mode="grayscale")

  ### val loss 기준 model save
  model_checkpoint = ModelCheckpoint(save_path, monitor='val_loss',verbose=1, save_best_only=True)
  train_generator=zip(image_generator, mask_generator)
  validation_generator=zip(image_validation_generator, mask_validation_generator)
  model.fit_generator(train_generator, steps_per_epoch=len(image_generator), epochs=epoch_number, validation_steps=len(image_validation_generator) , callbacks=[model_checkpoint,sc], validation_data=validation_generator)
  tf.keras.backend.clear_session()

if __name__ == '__main__':
  main()
