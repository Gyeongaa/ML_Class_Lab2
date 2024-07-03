# Import library
import os
import torch 
import torch.nn as nn 
import torch.backends.cudnn as cudnn 
import numpy as np 
import torchvision 
import pandas as pd 
from torchvision import datasets # models, transforms 
import matplotlib.pyplot as plt 
import time 
from torch.utils.data import DataLoader
from tqdm import tqdm 
import albumentations as A 
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import VisionDataset
from PIL import Image
from collections import Counter

cudnn.benchmark = True 
plt.ion() # 대화형 모드


# GPU 디바이스
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#################################################################
data_dir = '../test/tiles/'
output_dir = '../test/result/'
model_dir = '../model/'
#################################################################

files = os.listdir(data_dir)
filenames = [i[:-4] for i in files]

class CustomImageFolder(VisionDataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir 
        self.transform = transform
        self.images = sorted(os.listdir(root_dir))

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image
    

# Albumentations 변환 함수 정의 
class AlbumentationsTransform:
    def __init__(self, is_training=True):
        if is_training:
            self.transforms = A.Compose([
                A.Resize(224, 224), # 이미지 사이즈에 따라 달라짐
                A.Transpose(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.25),
                A.RandomBrightnessContrast(
                    brightness_limit = (-0.1, 0.1), 
                    contrast_limit = (-0.1, 0.1), p=0.25
                ), 
                A.Normalize([0.8158, 0.6012, 0.74041], [0.12153, 0.16785, 0.14023]),
                ToTensorV2(),
            ])
        else:
            self.transforms = A.Compose([
                A.Resize(224,224), 
                A.Normalize([0.8158, 0.6012, 0.74041], [0.12153, 0.16785, 0.14023]),
                ToTensorV2(),
            ])

    def __call__(self, img):
        return self.transforms(image=np.array(img))['image']
    

def model_prediction(test_loader, model):
    # Model prediction 
    y_pred = []

    for data in tqdm(test_loader):
        data = data.to(device)
        
        with torch.no_grad():
            output = model(data)
        
        # Prediction 
        _, pred = torch.max(output, 1)
        y_pred.extend(pred.cpu().numpy())
    
    return y_pred


### Load model 
model = torch.load(model_dir + 'classification_model.pth')

# Parameters
batch_size = 8
test_transforms = AlbumentationsTransform(is_training=False)

# Folders
# CustomImageDataset을 사용하여 테스트 데이터셋 생성 
test_dataset = CustomImageFolder(root_dir=data_dir, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

### Model prediction 
y_pred = model_prediction(test_loader, model)

df = pd.DataFrame(data=filenames, columns=['filename'])
df['prediction'] = y_pred
df['category'] = df['prediction'].map({0: 'benign', 1: 'normal', 2:'tumor'})


# Output folder 생성
os.makedirs(output_dir, exist_ok=True)

df.to_csv(output_dir + 'prediction.csv', index=False)

# Show inference result 
print(list(df['category']))