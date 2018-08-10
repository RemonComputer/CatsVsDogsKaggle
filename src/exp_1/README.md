# Readme
## Requirements
- Install Pandas python 
- This Exp builds upon previous Exp file: exp_0/main_1.py

## Goal
The Goal of the Exp is to setup the training and testing procedures and try basic deep learning architectures

## 0/
- I am desigining the basic network based on recommendation from https://www.youtube.com/watch?v=fTw3K8D5xDs
- I am also using https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py as my guide line
- This network is counter intutive because it has high number of features in lower layer and small number of features in higher layer which is counter intutive because as the spatial dimention increases the photo became more variable and the resolution can express higher details and shapes
- The network training, test, validation datasets accuracy was around 52% (after training for about 14 iterations) which clearly underfits due to the above point
- despite working on high resolution image (256x256x1) and having the lower layers has high number of features which increases the number of weights exponentially but one training epoch took about 82 seconds on GTX 1060
- I switched back to pycharm because VSCode debugging became teriminated for an unknown reason and this bug sometimes happen and sometime not 50% of the times
