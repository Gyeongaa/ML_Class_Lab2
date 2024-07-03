import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image
import warnings
from sklearn.model_selection import KFold
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing import image
import efficientnet.tfkeras as efn
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
import os
import pandas as pd
from PIL import Image
import warnings
from keras.layers import BatchNormalization,Add, MaxPooling3D, GlobalAveragePooling3D, Dense, Flatten,GlobalAveragePooling2D
from tensorflow.python.keras.applications.resnet import ResNet152
import tensorflow.keras as K
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import datasets, layers, models, losses, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import relu,softmax
from tensorflow.keras.models import Model
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.initializers import Zeros, Ones
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
from matplotlib import rcParams
from tqdm import tqdm
from glob import glob
import datetime
import time
warnings.filterwarnings("ignore")

print(os.getcwd())
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

def load_data(npy_path):
    print('-'*30)
    print('load images...')
    print('-'*30)
    
    whole = np.zeros((len(npy_path), 512, 512, 3))
    
    for i in tqdm(range(len(npy_path))):
        path = npy_path[i]
        train_npy_path = path
        imgs_tmp = np.array(Image.open(train_npy_path))
        whole[i] = imgs_tmp
        
            
    imgs_tmp = 0
    print('-'*30)
    print('imgs : {} '.format(whole.shape))     
    print('-'*30)
    imgs = whole.astype('float32')
    print('img : ', imgs.max())

    print('-'*30)
    print('normalization start...')
    print('-'*30)
    
    imgs = cv2.normalize(whole, None, 0, 1, cv2.NORM_MINMAX)
   
    print('img : ', imgs.max())

    return imgs

def main(model_path, test_csv_path):
    ###### Model prediction 시작 ######
    start = time.time()
    d = datetime.datetime.now()
    now_time = f"{d.year}-{d.month}-{d.day} {d.hour}:{d.minute}:{d.second}"
    print(f'[Model prediction Start]')
    print(f'Model prediction Start Time : {now_time}')

    model=load_model(model_path)
    print(model.summary())

    test = pd.read_csv(test_csv_path)
    test_x = load_data(test.path)

    test_y_before = test['class']
    GT = np.array(test_y_before)

    encoder = OneHotEncoder(sparse=False)
    test_y = encoder.fit_transform(test_y_before.values.reshape(-1, 1))

    predictions = model.predict(test_x, batch_size=30, verbose=1)
    predicted_class = np.argmax(predictions, axis=1)
    class_names = ['Normal', 'In situ', 'Malignant']

    cm = confusion_matrix(GT, predicted_class)
    print(cm)
    cm_df = pd.DataFrame(cm)
    cm_df.to_csv('./cm.csv', index=False)
    print(classification_report(GT, predicted_class, target_names = class_names))

    result_df = pd.DataFrame({'FileName': test['path'], 'GT': GT, 'Pred': predicted_class})
    result_df.to_csv('./result.csv', index=False)

    end = time.time()
    d = datetime.datetime.now()
    now_time = f"{d.year}-{d.month}-{d.day} {d.hour}:{d.minute}:{d.second}"
    print(f'Model prediction Time : {now_time}s Time taken : {end-start}')
    print(f'[Model prediction End]')
    ###### Model prediction 끝 ######