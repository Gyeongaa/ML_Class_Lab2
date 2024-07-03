import pandas as pd
from glob import glob
import Mask_preprocessing
import Tile_preprocessing
import label_csv
import prediction


# ##### Make mask #####
# normal_whi_list=glob('./data/Source_data/normal/WSI/*.tiff')
# insitu_whi_list=glob('./data/Source_data/insitu/WSI/*.tiff')
# malignant_whi_list=glob('./data/Source_data/malignant/WSI/*.tiff')

# normal_json_list=[f.replace('/WSI/', '/json/') for f in normal_whi_list]
# normal_json_list=[f.replace('.tiff', '.json') for f in normal_json_list]
# insitu_json_list=[f.replace('/WSI/', '/json/') for f in insitu_whi_list]
# insitu_json_list=[f.replace('.tiff', '.json') for f in insitu_json_list]
# malignant_json_list=[f.replace('/WSI/', '/json/') for f in malignant_whi_list]
# malignant_json_list=[f.replace('.tiff', '.json') for f in malignant_json_list]

# normal_mask_path='./data/mask_data/0_normal/'
# insitu_mask_path='./data/mask_data/1_insitu/'
# malignant_mask_path='./data/mask_data/2_malignant/'

# Mask_preprocessing.main(normal_whi_list, insitu_whi_list, malignant_whi_list, normal_mask_path, insitu_mask_path, malignant_mask_path, normal_json_list, insitu_json_list, malignant_json_list)



# ##### Make Tile #####
# normal_mask_files=glob('./data/mask_data/0_normal/*.tiff')
# insitu_mask_files=glob('./data/mask_data/1_insitu/*.tiff')
# malignant_mask_files=glob('./data/mask_data/2_malignant/*.tiff')

# normal_tile_path = './data/tile_data/0_normal/'
# insitu_tile_path = './data/tile_data/1_insitu/'
# malignant_tile_path = './data/tile_data/2_malignant/'

# Tile_preprocessing.main1(normal_whi_list, insitu_whi_list, malignant_whi_list, normal_mask_files, insitu_mask_files, malignant_mask_files, normal_tile_path, insitu_tile_path, malignant_tile_path)



# ##### Tile invert #####
# normal_tile_files=glob('./data/tile_data/0_normal/*.jpg')
# insitu_tile_files=glob('./data/tile_data/1_insitu/*.jpg')
# malignant_tile_files=glob('./data/tile_data/2_malignant/*.jpg')

# normal_invert_path = './data/tile_invert_data/0_normal/'
# insitu_invert_path = './data/tile_invert_data/1_insitu/'
# malignant_invert_path = './data/tile_invert_data/2_malignant/'

# Tile_preprocessing.main2(normal_tile_files, insitu_tile_files, malignant_tile_files, normal_invert_path, insitu_invert_path, malignant_invert_path)



##### Make test csv #####
normal_tile_invert_path = './data/tile_invert_data/0_normal/*.jpg'
insitu_tile_invert_path = './data/tile_invert_data/1_insitu/*.jpg'
malignant_tile_invert_path = './data/tile_invert_data/2_malignant/*.jpg'

csv_save_path = './data/csv/'

label_csv.main(normal_tile_invert_path, insitu_tile_invert_path, malignant_tile_invert_path, csv_save_path)




##### Model prediction #####
model_path = './Best_model.h5'
test_csv_path = './data/csv/test.csv'

prediction.main(model_path, test_csv_path)