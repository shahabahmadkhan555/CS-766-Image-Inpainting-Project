# Context-Aware Image Inpainting for Automatic Object Removal 

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fLbRK4v8gAwTNMn0_JT8cW2twOn1feD5?usp=sharing) <br>
The project webpage is available at https://nrussell11.github.io/CS-766-Image-Inpainting-Project/.
<-- -->
### Contents
1. [Project Demo](#project-demo)
2. [Installation and Setup](#installation-and-setup)
3. [Training](#training)
4. [Evaluation](#evaluation)

## Project Demo
To make it easier for people to quickly use this project without going through installation and setup steps, a self-contained Python notebook is available [here](https://colab.research.google.com/drive/1fLbRK4v8gAwTNMn0_JT8cW2twOn1feD5?usp=sharing).

## Installation and Setup
### Environment
The conda environment setup file `environment.yaml` is provided to install all necessary dependencies. The environment can be created and activated using the following command:
```bash
conda env create -f environment.yaml
conda activate image-inpainting-env
```

<details>
  <summary> Dependencies (click to expand) </summary>
  
  #### Dependencies
  - PyTorch 2.1
  - torchvision 0.16
  - torchmetrics
  - numpy
  - matplotlib
  - pillow
  - opencv-python
  - ultralytics

</details>

### Dataset
We used MIT's Miniplaces dataset for this project, which is a subset of the MIT Places dataset containing only 100,000 128x128 images across 100 scene categories. This dataset can be downloaded at [http://6.869.csail.mit.edu/fa17/miniplaces.html] or [https://github.com/CSAILVision/miniplaces].

Our code also supports any dataset with the following format:
```
├── some folder                                                                                                       
│   ├── ...                                                                                     
│                                                                                               
├── images                                                                                                                                                                                                       
│   ├── train                                                                                                  
│   │   └── category_1     
│   │   |   └── image1.jpg   
│   │   |   └── image2.jpg   
│   │   |   └── ...                                                                                                                   
│   │   └── category_2     
│   │   |   └── ...                                                                                
|   |   └── ...
|   ├── test
|   |   └── category_1 
|   |   └── category_2 
|   |   └── ...
|   ├── val
|   |   └── category_1 
|   |   └── category_1
|   |   └── ...
```
Note that the dataset does not have to have the exact structure. The code works by recursively going down sub-directories when given the root image directory, so you can simply put all images directly into training, test, and validation folders. Currently, only files with jpg extensions are supported, but this can be quickly changed in datasets.py. 
<!-- Files with extensions jpg, jpeg, and png are supported. -->

## Training
### Configuration
By default, the following configuration is used (details in train.py):
```
n_epochs: 40
batch_size: 16
lr: 0.0002
b1: 0.05
b2: 0.999
lambda_l1: 0.5
lambda_ssim: 0.5
latent_dim: 100
img_size: 128
mask_size: 64
channels: 3
```
---
To train the model:
```bash
python train.py --dataset_name {path_to_dataset}
```
---
To train the model from a checkpoint:
```bash
python train.py --dataset_name {path_to_dataset} --resume {path_to_checkpoint}
```
---
Example setting checkpoint interval (epochs between saving) and sample interval (batches between samples):
```bash
python train.py --dataset_name {path_to_dataset} --sample_interval {interval} --checkpoint_interval {interval}
```

## Evaluation
There are three different evaluation methods offered:
1. Evaluate on folders of images:
This image folder format works similarly to the dataset format described above.
```bash
python eval.py --model_checkpoint {path_to_checkpoint} --image_folder_path {path_to_image_folder}
```
By default, the batch size is set to the total number of images being evaluated, and the number of columns (images per row) is set to $\lceil \sqrt{\text{\\# images}} \rceil$ to get a square grid. To change this, run the following (this should be done if evaluating an entire dataset at once):
```bash
python eval.py --model_checkpoint {path_to_checkpoint} --image_folder_path {path_to_image_folder} --batch_size {size} --num_cols {num}
```
---
2. Evaluate on a single image:
```bash
python eval.py --model_checkpoint {path_to_checkpoint} --image_path {path_to_image}
```
---
3. Evaluate on a single image with automatic object removal:
A list of object classes that are supported by our automatic object removal is given in object_classes.txt. To remove a class of objects, simply add the id after '--remove'. For example, run the following to remove people (0) and cars (2):
```bash
python eval.py --model_checkpoint {path_to_checkpoint} --image_path {path_to_image} --remove 0 2
```
