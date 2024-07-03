# Import library
import os
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.optim import lr_scheduler 
import numpy as np 
import torchvision 
from torchvision import datasets, models, transforms 
import matplotlib.pyplot as plt 
import time 
import os 
import copy 
from torch.utils.data import DataLoader
from tqdm import tqdm 
import albumentations as A 
from albumentations.pytorch import ToTensorV2


#### Setting Parameters ####
EPOCH = 10
MODEL_NAME = 'classification_model.pth'
############################

#### Path ############################
data_dir = '../output/tiles_dataset'  # data
model_path = '../model/' # Model path
######################################

# GPU 디바이스
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 코드 시작 타임 
start_time = time.time()

# 난수 발생기의 시드 설정 
seed = 42 
torch.manual_seed(seed)
np.random.seed(seed)

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
                A.Normalize([0.8452, 0.5802, 0.7377], [0.1130, 0.1802, 0.1407]),
                ToTensorV2(),
            ])
        else:
            self.transforms = A.Compose([
                A.Resize(224,224), 
                A.Normalize([0.8452, 0.5802, 0.7377], [0.1130, 0.1802, 0.1407]),
                ToTensorV2(),
            ])

    def __call__(self, img):
        return self.transforms(image=np.array(img))['image']
    

# Albumentations 변환 클래스 생성
data_transforms = {
    'train': AlbumentationsTransform(is_training=True), 
    'valid' : AlbumentationsTransform(is_training=False),
    'test': AlbumentationsTransform(is_training=False),
}

# Dataset root 
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train','valid']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=8, pin_memory=True) for x in ['train','valid']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train','valid']}
class_names = image_datasets['train'].classes 


# 모델 학습하기 
def train_model(model, criterion, optimizer, scheduler, num_epochs=EPOCH):
    """
    Pytorch를 활용한 모델 학습 코드 

    Input: Dataloader, model parameters 
    Output: 성능이 가장 좋은 학습된 모델의 가중치
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-'*10)

        # 각 에폭은 학습 단계와 검증 단계를 갖습니다. 
        for phase in ['train','valid']:
            if phase == 'train':
                model.train() # 모델을 학습 모드로 설정 
            else:
                model.eval() # 모델을 평가 모드로 설정 

            running_loss = 0.0 
            running_corrects = 0

            # 데이터를 반복 
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 매개변수 경사도를 0으로 설정 
                optimizer.zero_grad()

                # 순전파. 학습 시에만 연산 기록을 추적 
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 학습 단계인 경우 역전파 + 최적화 
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # 통계 
                running_loss += loss.item()*inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase}Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 모델을 deep copy 함 
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
    
    time_elapsed = time.time() - since 
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    # 가장 나은 모델 가중치를 불러옴
    model.load_state_dict(best_model_wts)
    
    return model

# Execution 
# 고정된 특징 추출기로써의 합성곱 신경망 
model_conv = torchvision.models.resnet18(weights='IMAGENET1K_V1')

for param in model_conv.parameters():
    param.requires_grad = False

# 새로 생성된 모듈의 매개변수는 기본값이 requires_grad = True
num_ftrs = model_conv.fc.in_features

# 모델 마지막 FC 레이어 층 변경 
model_conv.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512), 
    nn.ReLU(), 
    nn.Dropout(0.3),
    nn.Linear(512, len(class_names))
)

criterion = nn.CrossEntropyLoss()

# 이전과는 다르게 마지막 계층의 매개변수들만 최적화되는지 관찰 
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# 5에폭마다 0.1씩 학습률 감소 
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=5, gamma=0.1)

# GPU 병렬 처리 
model_conv = nn.DataParallel(model_conv)
model_conv.to(device)


# 학습 및 평가하기 
model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=EPOCH)

os.makedirs(model_path, exist_ok=True)

# 모델명 설정
model_name = MODEL_NAME
torch.save(model_conv, model_path + model_name)

# Time 
end_time = time.time()
execution_time = end_time - start_time 

print(f"코드 실행 시간: {execution_time}초")