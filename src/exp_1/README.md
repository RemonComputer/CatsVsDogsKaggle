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
- After trying to increase the learning rate (see 1/) the network accuracies became: train set accuracy: 77.44%, test set accuracy: 76.8%, validation set accuracy: 77.12%

## 1/
- After trying the intitive design of the neural network, The same problem happened (accuracy about 52%)
- After some debugging and thinking I saw that the outputs of the network didn't change much (all of them was around 0.5), So it is either a learning rate problem or a vanishing gradient problem (My network isn't that deep)
- So I increased the learning rate from 0.001 to 0.01, So the network accuracy on the training dataset jumped to about 98%, So I concluded it was a learning rate problem.
- What stopped me from understanding that it was a learning rate problem was that the accuracy on patches kept jumping back and forth, And I assumed that when a learning rate problem happens is that the accuracy should increase by small amount every time but it was jumping because I am using mini-batchs not all the dataset so the result was a little bit noisy
- I think the previous network had the same issue (I solved the issue see 0/ for details)
- Final Trainig set accuracy: 98.995%, Final Test set accuracy: 88.32%, Final validation set accuracy: 89.52% So clearly the network over fits and need some form of regularization or smaller number of parameters 

## 2/
- Still the network overfits even after penalizing it with large L1 Regularization Criterion
- After training it for 50 epochs with batch size = 128 and sparse factor = 1e-1, I obtained: train set accuracy: 99.995%, test set accuracy: 85.76%, validation set accuracy: 86.76

## 3/
- Added Randomization before selecting the training, test, validation sets because I suspected the nature of the train, test, validation sets wasn't the same, This is bacuse it seems that number of misclassifed images in the train, test, validation to be 480 image
- Increased the learning rate from 0.01 to 0.1 because the network seemed not to learn after randomization of the datasets
- Tried the data augmentation to increase the accuracy of the network and it worked
- After traininig on the augmented training set for 50 epochs and 10 epochs on non-augemted dataset I obtained: train set accuracy: 95.865%, test set accuracy: 95.44%, validation set accuracy: 95.08%

## 4/
- Tried Advanced Optimizers like Adam at an approperiate learning rate 1e-4 it seams mostly like it takes steady steps towards convergence. 
- Tried using the color channels but multiplied the wights by 3 .
- Trained the network for 8 hours and 16 minutes for 107 iterations.
- At the end the performance of the network was vibrating from 95% to 98% on the test batches so I added a few lines to quit training if the test accuracy passed 98%.
- Finaly I obtained a performance of: train set accuracy: 99.835%, test set accuracy: 97.24%, validation set accuracy: 97.28%
- You can find the Pre-Trained model at: https://drive.google.com/file/d/1RGW67-YOb2yFQfgCpvMEgrdzJd2r6rl7/view?usp=sharing 
