import pandas as pd
import data_preprocess
import prediction
from glob import glob
from torchvision.transforms.functional import to_pil_image
import torch
import torch.nn.functional as F
import os
import time
import datetime
import numpy as np


def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")


def dice_loss(pred, target, num_classes):
    smooth = 2
    dice_per_class = torch.zeros(num_classes).to(pred.device)

    for class_id in range(num_classes):
        pred_class = pred[class_id,  ...]
        target_class = target[class_id,  ...]

        intersection = torch.sum(pred_class * target_class)
        A_sum = torch.sum(pred_class * pred_class)
        B_sum = torch.sum(target_class * target_class)

        dice_per_class[class_id] = 1 - \
            (2. * intersection + smooth) / (A_sum + B_sum + smooth)

    return torch.mean(dice_per_class)


start = time.time()
d = datetime.datetime.now()
now_time = f"{d.year}-{d.month}-{d.day} {d.hour}:{d.minute}:{d.second}"
print(f'[sess Start]')
print(f'sess Start Time : {now_time}')
whi_list = glob('./data/Source_data/**/WSI/*.tiff')
label_data = pd.read_csv('./test.csv')
image_path = './data/preprocessing/image/'
normal_mask_path = './data/preprocessing/polygon/NT_normal/'
tumor_mask_path = './data/preprocessing/polygon/TP_tumor/'
data_preprocess.Preprocessing(
    whi_list, image_path, normal_mask_path, tumor_mask_path)
image_list = glob('./data/preprocessing/image/*.tiff')
total_path, total_y, total_prob, total_dice = prediction.Predict(image_list)
label_data['dice'] = 0
tumor_pre_mask = total_prob[:, 1]
normal_pre_mask = total_prob[:, 2]
tumor_mask = total_y[:, 1]
normal_mask = total_y[:, 2]
for i in range(len(tumor_mask)):
    if np.array(label_data.loc[label_data['FileName'] == total_path[i][0], ['Class']])[0, 0] == 'normal':
        label_data.loc[label_data['FileName'] == total_path[i][0], ['dice']] = float(
            int((1-dice_loss(normal_pre_mask[i], normal_mask[i], 1).item())*10000))/100
        createDirectory('./data/predict/'+total_path[i][0])
        to_pil_image((normal_pre_mask[i].type(
            torch.uint8))*255).save('./data/predict/'+total_path[i][0]+'/pred_NT_mask.jpg')
        to_pil_image((normal_mask[i].type(
            torch.uint8))*255).save('./data/predict/'+total_path[i][0]+'/NT_GT.jpg')
    else:
        label_data.loc[label_data['FileName'] == total_path[i][0], ['dice']] = (float(int(
            (1-dice_loss(normal_pre_mask[i], normal_mask[i], 1).item())*10000))/100+float(int((1-dice_loss(tumor_pre_mask[i], tumor_mask[i], 1).item())*10000))/100)/2
        createDirectory('./data/predict/'+total_path[i][0])
        to_pil_image((tumor_pre_mask[i].type(
            torch.uint8))*255).save('./data/predict/'+total_path[i][0]+'/pred_TP_mask.jpg')
        to_pil_image((normal_pre_mask[i].type(
            torch.uint8))*255).save('./data/predict/'+total_path[i][0]+'/pred_NT_mask.jpg')
        to_pil_image((tumor_mask[i].type(
            torch.uint8))*255).save('./data/predict/'+total_path[i][0]+'/TP_GT.jpg')
        to_pil_image((normal_mask[i].type(
            torch.uint8))*255).save('./data/predict/'+total_path[i][0]+'/NT_GT.jpg')
label_data.to_csv('./data/predict_label.csv', index=False)
Dice_score = label_data['dice'].mean()
print(f'Dice-coefficient={Dice_score}')
end = time.time()
d = datetime.datetime.now()
now_time = f"{d.year}-{d.month}-{d.day} {d.hour}:{d.minute}:{d.second}"
print(f'sess Time : {now_time}s Time taken : {end-start}')
print(f'[sess End]')
