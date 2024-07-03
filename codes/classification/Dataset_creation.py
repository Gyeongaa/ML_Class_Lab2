# Import Library
import os 
import openslide
import numpy as np
import json 
import glob
import cv2 
import matplotlib.pyplot as plt
import time 


#### Parameter Settings ####
tile_size = 224
scale = 8.0
#############################

###### Path Settings ############
img_path = '../data/wsi/'
anno_path = '../data/annotation/'
mask_path = '../output/mask/'
output_path = '../output/tiles/'
#################################


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
    
def tiff_to_image(file, folder, anno_path, img_path, scale):
    
    # Assign file path
    mask_file = os.path.join(anno_path, folder, file) + '.json'
    tiff_file = os.path.join(img_path, folder, file) +'.tiff'

    # Load json file
    with open(mask_file) as json_file:
        annotations = json.load(json_file)

     # Extract Json's obejct 
    obj = annotations['files'][0]['objects']

    # Load scale downed image
    image = scale_down_wsi(tiff_file, scale)

    benign_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    normal_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    tumor_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    benign_roi = None
    normal_roi = None
    tumor_roi = None

    for annotation in obj:
        label = annotation['label']

        if label == 'TP_benign':
            coordinates = np.asarray(annotation['coordinate'], dtype=np.int32)
            modified = np.asarray([[int(x // float(scale)), int(y // float(scale))] for x, y in coordinates], dtype=np.int32)
            cv2.fillPoly(benign_mask, [modified], 255)
            benign_roi = cv2.bitwise_and(image, image, mask=benign_mask)
            # 0 인 지점을 모두 255 (흰색 바탕으로 치환)
            benign_roi[benign_mask == 0] = 255.

        elif (label == 'NT_normal_renal') | (label == 'NT_normal_extrarenal'):
            coordinates = np.asarray(annotation['coordinate'], dtype=np.int32)
            modified = np.asarray([[int(x // float(scale)), int(y // float(scale))] for x, y in coordinates], dtype=np.int32)
            cv2.fillPoly(normal_mask, [modified], 255)
            normal_roi = cv2.bitwise_and(image, image, mask=normal_mask)
            normal_roi[normal_mask == 0] = 255.

        elif label == 'TP_malignant':
            coordinates = np.asarray(annotation['coordinate'], dtype=np.int32)
            modified = np.asarray([[int(x // float(scale)), int(y // float(scale))] for x, y in coordinates], dtype=np.int32)
            cv2.fillPoly(tumor_mask, [modified], 255)
            tumor_roi = cv2.bitwise_and(image, image, mask=tumor_mask)
            tumor_roi[tumor_mask == 0] = 255.
        else:
            pass

    return benign_roi, normal_roi, tumor_roi


def number_of_tiles(image, tile_size):
    height, width, _ = image.shape
    num_tiles_x = width // tile_size 
    num_tiles_y = height // tile_size 

    return num_tiles_x, num_tiles_y


def create_tiles(file, image, class_folder, output_path, tile_size): 
    folder = class_folder
    num_tiles_x, num_tiles_y = number_of_tiles(image, tile_size)

    # Create tiles 
    for x in range(num_tiles_x):
        for y in range(num_tiles_y):
            tile = image[y*tile_size: (y+1)*tile_size, x*tile_size: (x+1)*tile_size]
            # 각 타일의 흰 여백 체크 
            ratio_of_255 = np.mean(tile == 255)
            if ratio_of_255 > 0.75:
                pass 
            else:
                # File path
                target_path = os.path.join(output_path, folder)
                # os.makedirs(target_path, exist_ok=True)
                plt.imsave(target_path + '/' + file + '_' + str(y) + '_' + str(x) + '.jpg', tile)

########################################################################################################

if __name__ == '__main__':
    # Start time
    start_time = time.time()
    folders = ['benign','normal','tumor']

    # 폴더 별 파일명 읽기
    for folder in folders:

        # Whole list
        images = glob.glob(img_path + folder + '/' + '*')
        masks = glob.glob(anno_path + folder + '/' + '*')

        # Check for common file name
        files_img = [i.split('/')[-1][:-5] for i in images]
        files_mask = [i.split('/')[-1][:-5] for i in masks]

        # img와 annotation 맵핑이 되는 경우만 ID 출력
        common_files = list(set(files_img) & set(files_mask))
        common_files.sort()  # 알파벳 순으로 정렬

        # Default
        total_class = ['benign','normal','tumor']
        total_image = None

        # Create output folder
        for f in total_class:
            os.makedirs(output_path + f, exist_ok=True)
            os.makedirs(mask_path + f, exist_ok=True)

        # Create mask and tile images
        for file in common_files:
            # Tiff to image
            benign, normal, tumor = tiff_to_image(file, folder, anno_path, img_path, scale)

            # If you want to save mask images 
            
            total_image = [benign, normal, tumor]

            for idx, img in enumerate(total_image):
                if (img is not None) & (idx == 0):
                    # If you want to save mask images 
                    target_path = os.path.join(mask_path, 'benign')
                    cv2.imwrite(target_path +'/' + file + '.jpg', img)

                    # Create tiles 
                    create_tiles(file, img, 'benign', output_path, tile_size)
                
                elif (img is not None) & (idx == 1):
                    target_path = os.path.join(mask_path, 'normal')
                    cv2.imwrite(target_path + '/' + file + '.jpg', img)

                    # Create tiles 
                    create_tiles(file, img, 'normal', output_path, tile_size)

                elif (img is not None) & (idx == 2):
                    target_path = os.path.join(mask_path, 'tumor')
                    cv2.imwrite(target_path + '/' + file + '.jpg', img)

                    # Create tiles 
                    create_tiles(file, img, 'tumor', output_path, tile_size)
                else:
                    pass
            
            print(f'{file} tiling process is done!')

    # End time 
    end_time = time.time()
    execution_time = (end_time - start_time) // 60
    print('-'*50)
    print(f'Data creation process took {execution_time} minutes!')