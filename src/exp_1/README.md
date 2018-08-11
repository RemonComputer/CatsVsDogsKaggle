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

## 1/
- After trying the intitive design of the neural network, The same problem happened (accuracy about 52%)
- After some debugging and thinking I saw that the outputs of the network didn't change much (all of them was around 0.5), So it is either a learning rate problem or a vanishing gradient problem (My network isn't that deep)
- So I increased the learning rate from 0.001 to 0.01, So the network accuracy on the training dataset jumped to about 98%, So I concluded it was a learning rate problem.
- What stopped me from understanding that it was a learning rate problem was that the accuracy on patches kept jumping back and forth, And I assumed that when a learning rate problem happens is that the accuracy should increase by small amount every time but it was jumping because I am using mini-batchs not all the dataset so the result was a little bit noisy
- I think the previous network had the same issue
- Final Trainig set accuracy: 98.995%, Final Test set accuracy: 88.32%, Final validation set accuracy: 89.52% So clearly the network over fits and need some form of regularization or smaller number of parameters 
