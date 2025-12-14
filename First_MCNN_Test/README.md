# MCNN-pytorch
This is an simple and clean implemention of CVPR 2016 paper ["Single-Image Crowd Counting via Multi-Column Convolutional Neural Network."](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf)  
# Installation
&emsp;1. Create a Python environment using `Python -m venv venv`
&emsp;2. Activate the environment using "source venv/bin/activate"
&emsp;3. Run `pip install -r requirements.txt`

&emsp;4. Clone this repository  
```git
git clone https://github.com/joey11121/DIP_Final.git
```
We'll call the directory that you cloned MCNN-pytorch as ROOT.
# Data Setup
&emsp;1. Download ShanghaiTech Dataset from my Google Drive
Google Drive: [link](https://drive.google.com/file/d/1zvsRJffj0zYre5fjQ5nBlo15L8Gn_BTe/view?usp=sharing)
&emsp;2. Extract and put ShanghaiTech Dataset in ROOT.  Use "data_preparation/k_nearest_gaussian_kernel.py" to generate ground truth density-map. (Mind that you need modify the root_path in the main function of "data_preparation/k_nearest_gaussian_kernel.py, and I have created the density map for part_A. You can generate the density maps for the images in part_B if you want.")  
# Training
## Original Code
&emsp;1. Modify the root path in "train.py" according to your dataset position.   
&emsp;2. In command line: Run `python train.py` for the original default code. 

## Modified Code 
Run `python train_plus.py` in the command line. 

# Testing
## Testing The Model Trained From Original Code
&emsp;1. Modify the root path in "test.py" according to your dataset position.  
&emsp;2. Run test.py for calculate MAE of test images or just show an estimated density-map.  

## Testing the Model Trained From Modified Code 
&emsp;1. Modify the root path in "test_plus.py" according to your dataset position.  
&emsp;2. Run test_plus.py for calculate MAE of test images or just show an estimated density-map.  
# Other notes
&emsp;1. Unlike original paper, this implemention doesn't crop patches for training. We directly use original images to train mcnn model and also achieve the result as authors showed in the paper.  
&emsp;2. If you are new to crowd counting, we recommand you to know [Crowd_counting_from_scratch](https://github.com/CommissarMa/Crowd_counting_from_scratch) first. It is an overview and tutorial of crowd counting.
