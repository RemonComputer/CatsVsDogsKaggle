# Readme
## Requirements
you need to install: 
scikit-image, torch, torchvision, numpy, matplotlib, cudasdk (if you have Nvidia Combitable GPU),
VSCode as ide and configure the interpreter

## Goal
The Goal of the Exp is to make the dataloading and basic transformations so that the data is ready for neural network training

## main_0.py
### Intutive understanding
I choose the input image size to be 256 because 256 is a power of 2 for the ease of computation, despite the fact that 256 is too high this is because I tried to see some images with other low resolutions like 64 (see https://www.kaggle.com/jeffd23/catdognet-keras-convnet-starter) but the images didn't show the details and it was hard for me as a human being to recognize so what about the machine.

I normalized from 0 to 1 instead of -1 to 1 this is because I will use Relu which has an operating range of 0 to infinity so it will clip the negative side which will has half of the data, maybe we don't need normalization (but we might need the convertion to float data type).

I choose Inter cubic interpolation in the image resizing algorithm because it did a good job at lower resolution than skimage.transform.resize and it was suggested at https://www.kaggle.com/jeffd23/catdognet-keras-convnet-starter

## main_1.py
### What has been changed
- I am using the implemented transformations in torchvision package (standarization of the code) -> see https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
- Trying the Grayscale image transformation.
- Supporting of image showing with grayscale.
- Trying normalization in the range of -1 to 1 values.
- Trying rescaling with the default bilinear transformation it shouldn't matter alot with high resolution images.
- Trying to load the data with the dataloader.
- Trying to iterate on the data using python iterators.
- Dislaying the class label with the above pytorch tutorial.
- Using manual seed to replicate results across multiple runs.