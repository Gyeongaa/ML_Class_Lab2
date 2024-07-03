# Import
import os 
import openslide
import numpy as np
import glob
import cv2 
import matplotlib.pyplot as plt
import time 
import scipy.ndimage as ndi
from skimage import io
from PIL import Image 

############## Path ################
img_path = '../test/wsi/'
output_img_path = '../test/image/'
output_mask_path = '../test/mask/'
output_tile_path = '../test/tiles/'
####################################

# Create folders 
os.makedirs(output_img_path, exist_ok=True)
os.makedirs(output_mask_path, exist_ok=True)
os.makedirs(output_tile_path, exist_ok=True)

# Parameters 
data_parameters = {'tile_size': 224, 
                   'scale': 8.0,
                   'mask_size': 256}

def scale_down_wsi(wsi_image, scale):
    
    slide = openslide.open_slide(wsi_image)
    try:
        level = slide.level_downsamples.index(scale)
        new_dimension = slide.level_dimensions[level]
        return np.asarray(slide.get_thumbnail(new_dimension))
    
    except:  # Level이 딱 8.0과 같이 정수로 안떨어지는 경우가 있음
        possible_level = slide.level_downsamples
        possible_elements = [int(i - scale) for i in possible_level]
        level = possible_elements.index(0)
        new_dimension = slide.level_dimensions[level]
        return np.asarray(slide.get_thumbnail(new_dimension))
    

def tiff_to_image(file, img_path, scale):
    
    # Assign file path
    tiff_file = os.path.join(img_path, file) +'.tiff'

    # Load scale downed image
    image = scale_down_wsi(tiff_file, scale)
        
    return image

def get_masked_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    _, a, _ = cv2.split(lab)
    th = cv2.threshold(a, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    mask = np.zeros_like(a)
    mask[a < th] = 1
    mask[a >= th] = 2
    mask = ndi.binary_fill_holes(mask-1)

    masked_image = np.zeros_like(image)
    masked_image[mask == 1] = image[np.where(mask == 1)]
    masked_image[mask == 0] = 255. 

    return masked_image

def create_tiles(image, num_tiles_x, num_tiles_y, file, output_path, tile_size):
    # Create tiles 
    for x in range(num_tiles_x):
        for y in range(num_tiles_y):
            tile = image[y*tile_size: (y+1)*tile_size, x*tile_size: (x+1)*tile_size]

            # 각 타일이 흰 여백 체크 
            ratio_of_255 = np.mean(tile == 255)
            if ratio_of_255 <= 0.25:
                plt.imsave(output_path + '/' + file + '_' + str(y) + '_' + str(x) + '.jpg', tile)
            else:
                pass

###################### Whole list ##########################
images = glob.glob(img_path + '*')
common_files = [i.split('/')[-1][:-5] for i in images]
common_files.sort()

# Create mask and tile images
for file in common_files:
    # Tiff to image
    image = tiff_to_image(file, img_path, data_parameters['scale'])
    # 노이즈를 제거한 깨끗한 원본 이미지
    masked_image = get_masked_image(image)

    # 원본, 마스크 이미지 저장
    plt.imsave(output_mask_path + file + '.jpg', masked_image)
   
    print(f'{file} mask image is created!')
##############################################################


##################### Create tiles ###########################
# Create mask and tile images
for file in common_files:

   # Load images
   image = io.imread(output_mask_path + file + '.jpg')
   height, width, _ = image.shape 
   num_tiles_x = width // 224
   num_tiles_y = height // 224

   # Create tile images
   create_tiles(image, num_tiles_x, num_tiles_y, file, output_tile_path, data_parameters['tile_size'])

   print(f'{file} is created!')

##################################################################