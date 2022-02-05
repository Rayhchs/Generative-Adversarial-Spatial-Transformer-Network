# Generative-Adversarial-Spatial-Transformer-Network (GASTN)

[Spatial Transformer Network](https://arxiv.org/pdf/1506.02025.pdf), proposed by Max Jaderberg, Karen Simonyan, Andrew Zisserman and Koray Kavukcuoglu, is used to solve the spatial invariant problem of CNN model. In the origin paper, STN is a module which can be inserted anywhere in the network. Therefore, this repository consturcts a Generative Adversarial Network using Spatial Transformer Network to detect and orthorify the object from an image. In this repo, we use this GASTN to detect and orthorify the license plate of Taiwan from an image. However, due to lack of car plate images, this repo only adopts approximately 100 of training data and several testing data.

## Spatial Transformer Network
<div align=center><img src="https://github.com/Rayhchs/Generative-Adversarial-Spatial-Transformer-Network/blob/main/img/STN.png" width="720"></div>

Spatial transformer network is composed of 3 elements:
* **Localization network (LN)**:
Localization network takes the image or feature map as input and output 6 parameters. These 6 parameters will be used as affine transformation of the input image (or feature map).

* **Grid generator**:
Grid generator generate a new grid of x,y coordinate from input image (or feature map) via affine transformation where the parameters are provided by localization network.

* **Sampler**:
Since the output parameters of localization network is not usually integers, sampler is used to interpolate pixel value from input image (or feature map) and produce new value for new grid generated by grid generator.

## Requisite

* python 3.8
* tensorflow 2.5.0
* Cuda 11.1
* CuDNN 8.1.1

## Network Structure
<div align=center><img src="https://github.com/Rayhchs/Generative-Adversarial-Spatial-Transformer-Network/blob/main/img/Network.jpg" width="720"></div>
The network is composed of a STN generator and a discriminator. The STN generator is used for spatial transformation of the input image. The discriminator distinguishes the orthorified image and non-orthorified image.

### STN Generator
Generator is a single STN. This repo only modify the localization net of STN, which is shown in below:
<div align=center><img src="https://github.com/Rayhchs/Generative-Adversarial-Spatial-Transformer-Network/blob/main/img/LN.jpg" width="600"></div>

### Discriminator
Discriminator follows the structure of [PatchGAN](https://arxiv.org/pdf/1611.07004v3.pdf)

### Objective
For Generator loss, this repository involves Vanilla GAN loss as well as MSE loss. The discriminator is going to maximize GAN loss and the generator is going to minimize summation of GAN loss and MSE loss. The final objective of Generator is described in below.

* **Vanilla GAN loss:**  
<div align=center><img src="https://github.com/Rayhchs/Generative-Adversarial-Spatial-Transformer-Network/blob/main/img/GAN_loss.png" height="48"></div>

* **L2 loss:**  
<div align=center><img src="https://github.com/Rayhchs/Generative-Adversarial-Spatial-Transformer-Network/blob/main/img/MSE_loss.png" height="60"></div>
Where **y** is orthorified image and **G(x)** is transformed image

* **Final objective:** 
<div align=center><img src="https://github.com/Rayhchs/Generative-Adversarial-Spatial-Transformer-Network/blob/main/img/OB.png" height="72"></div>
**Lambda** is set as 1 for this case.

## Getting Started
* Clone this repository

      git clone https://github.com/Rayhchs/Generative-Adversarial-Spatial-Transformer-Network.git
      cd Generative-Adversarial-Spatial-Transformer-Network
      
* Clone STN module from [spatial-transformer-network](https://github.com/kevinzakka/spatial-transformer-network)

      git clone https://github.com/kevinzakka/spatial-transformer-network.git
    
* Train

	  python -m main train <txt filename>
	  
Txt file will contain filename of input and label images. (e.g., ./data/train/A/1.jpg, ./data/train/B/1.jpg)

* For training detail:

	  tensorboard --logdir=./log

* Test

	  python -m main test
	  
There is one input: direction of testing image

## Results
The model is trained by less than 100 data. The config.py shows the configuration of the training. Testing results are shown in below.
| Origin image | Transformed image | Ground truth |
| ------------- | ------------- | ------------- |
| <img src="https://github.com/Rayhchs/Generative-Adversarial-Spatial-Transformer-Network/blob/main/data/test/1387089134-2970851851.jpg" width="250"> | <img src="https://github.com/Rayhchs/Generative-Adversarial-Spatial-Transformer-Network/blob/main/result/1387089134-2970851851.jpg" width="250"> | <img src="https://github.com/Rayhchs/Generative-Adversarial-Spatial-Transformer-Network/blob/main/data/true/1387089134-2970851851.jpg" width="250">|
| <img src="https://github.com/Rayhchs/Generative-Adversarial-Spatial-Transformer-Network/blob/main/data/test/278132117_274e46f759.jpg" width="250"> | <img src="https://github.com/Rayhchs/Generative-Adversarial-Spatial-Transformer-Network/blob/main/result/278132117_274e46f759.jpg" width="250"> | <img src="https://github.com/Rayhchs/Generative-Adversarial-Spatial-Transformer-Network/blob/main/data/true/278132117_274e46f759.jpg" width="250">|
| <img src="https://github.com/Rayhchs/Generative-Adversarial-Spatial-Transformer-Network/blob/main/data/test/gm-didi-app-lets-you-scan-a-license-plate-and-text-the-owner-wait-what-82759-7.jpg" width="250"> | <img src="https://github.com/Rayhchs/Generative-Adversarial-Spatial-Transformer-Network/blob/main/result/gm-didi-app-lets-you-scan-a-license-plate-and-text-the-owner-wait-what-82759-7.jpg" width="250"> | <img src="https://github.com/Rayhchs/Generative-Adversarial-Spatial-Transformer-Network/blob/main/data/true/gm-didi-app-lets-you-scan-a-license-plate-and-text-the-owner-wait-what-82759-7.jpg" width="250">|
| <img src="https://github.com/Rayhchs/Generative-Adversarial-Spatial-Transformer-Network/blob/main/data/test/images%20(1).jpg" width="250"> | <img src="https://github.com/Rayhchs/Generative-Adversarial-Spatial-Transformer-Network/blob/main/result/images%20(1).jpg" width="250"> | <img src="https://github.com/Rayhchs/Generative-Adversarial-Spatial-Transformer-Network/blob/main/data/true/images%20(1).jpg" width="250"> |
| <img src="https://github.com/Rayhchs/Generative-Adversarial-Spatial-Transformer-Network/blob/main/data/test/images.jpg" width="250"> | <img src="https://github.com/Rayhchs/Generative-Adversarial-Spatial-Transformer-Network/blob/main/result/images.jpg" width="250"> | <img src="https://github.com/Rayhchs/Generative-Adversarial-Spatial-Transformer-Network/blob/main/data/true/images.jpg" width="250">|

| Aerial image | Generated map | Ground truth |
| ------------- | ------------- | ------------- |
| <img src="https://github.com/Rayhchs/Pix2pix-tensorflow-implementation/blob/main/test/3.jpg" width="250"> | <img src="https://github.com/Rayhchs/Pix2pix-tensorflow-implementation/blob/main/result/3.jpg" width="250"> | <img src="https://github.com/Rayhchs/Pix2pix-tensorflow-implementation/blob/main/test/label_3.jpg" width="250">|
| <img src="https://github.com/Rayhchs/Pix2pix-tensorflow-implementation/blob/main/test/3.jpg" width="250"> | <img src="https://github.com/Rayhchs/Pix2pix-tensorflow-implementation/blob/main/result/3.jpg" width="250"> | <img src="https://github.com/Rayhchs/Pix2pix-tensorflow-implementation/blob/main/test/label_3.jpg" width="250">|
| <img src="https://github.com/Rayhchs/Pix2pix-tensorflow-implementation/blob/main/test/3.jpg" width="250"> | <img src="https://github.com/Rayhchs/Pix2pix-tensorflow-implementation/blob/main/result/3.jpg" width="250"> | <img src="https://github.com/Rayhchs/Pix2pix-tensorflow-implementation/blob/main/test/label_3.jpg" width="250">|
| <img src="https://github.com/Rayhchs/Pix2pix-tensorflow-implementation/blob/main/test/3.jpg" width="250"> | <img src="https://github.com/Rayhchs/Pix2pix-tensorflow-implementation/blob/main/result/3.jpg" width="250"> | <img src="https://github.com/Rayhchs/Pix2pix-tensorflow-implementation/blob/main/test/label_3.jpg" width="250">|
| <img src="https://github.com/Rayhchs/Pix2pix-tensorflow-implementation/blob/main/test/3.jpg" width="250"> | <img src="https://github.com/Rayhchs/Pix2pix-tensorflow-implementation/blob/main/result/3.jpg" width="250"> | <img src="https://github.com/Rayhchs/Pix2pix-tensorflow-implementation/blob/main/test/label_3.jpg" width="250">|
## Acknowledge
STN code is borrowed from [spatial-transformer-network](https://github.com/kevinzakka/spatial-transformer-network). Thanks for the excellent work!
