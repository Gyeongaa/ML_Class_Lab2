## Development of segmentation/classification model  using kidney cancer pathology images


### Segmentation Model Description

 - Purpose: Display normal, benign, and malignant tumor segments (Segmentaiton)
 - Model architecture: U-Net (efficientb0 backbone)
 - Input: Image with completed data preprocessing
 - Output: Mask image with area marked
 - training_dataset: 10,000 WSI (80% of total), image size adjusted 1,024
 - training element: 
    a. Loss function: bce_jaccard_loss 
    b. Optimizer: Adam 
    c. Epoch: 120 (default)
    d. Learning rate: 1e-4
    e. Batch size: 1 
    f. Evaluation metric: iou_score
   
#### 1. Host environments 

- OS==Windows 11 Home
- RAM==32 GB
- GPU== NVIDIA Geforce RTX 4070
- Storage== 3TB
- Docker==20.10.21

#### 2. Required Libraries 

- segmentation-models==1.0.1
- numpy==1.26.1
- openslide-python==1.3.1
- opencv-python==4.8.1.78
- jsons==1.6.3
- pandas==2.1.4
- scikit-learn==1.3.2
