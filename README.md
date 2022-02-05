# Generative-Adversarial-Spatial-Transformer-Network (GASTN)

[Spatial Transformer Network](https://arxiv.org/pdf/1506.02025.pdf), proposed by Max Jaderberg, Karen Simonyan, Andrew Zisserman and Koray Kavukcuoglu, is used to solve the spatial invariant problem of CNN model. In the origin paper, STN is a module which can be inserted anywhere in the network. Therefore, this repository consturcts a Generative Adversarial Network using Spatial Transformer Network to detect and orthorify the object from an image. In this repo, we use this GASTN to detect and orthorify the license plate of Taiwan from an image. However, due to lack of car plate images, this repo only adopts approximately 100 of training data and several testing data.

## Spatial Transformer Network
Spatial transformer network is composed of 3 elements:
* Localization network (LN):
Localization network takes the image or feature map as input and output 6 parameters. These 6 parameters will be used as affine transformation of the input image (or feature map).

* Grid generator:
Grid generator generate a new grid of x,y coordinate from input image (or feature map) via affine transformation where the parameters are provided by localization network.

* Sampler:


