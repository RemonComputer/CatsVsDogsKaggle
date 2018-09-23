# Readme
## Requirements
- The same Requirements of https://github.com/RemonComputer/CatsVsDogsKaggle/blob/master/src/exp_1/README.md
- This Exp builds upon previous Exp file: exp_1/4/main.py

## Goal
The Goal of the Exp is to try advanced techniques and advanced architectures.

## 0/
- This is the First experiment of the transfer learning experiments.
- Trying using ResNet18 pre-trained model, replacing the last layer with the layer that calculates the propability of being a dog.
- All the pre-trained model parameters are fixed except the new layer parameters.  
- Modified the validation to be on the whole test set.
- Modified the selection of the model parameters to be the best parameters parameters that has smallest validation loss.
- Trained the model for  115 iterations (1h 33 mins).
- train set accuracy: 98.12%, train set average loss: 0.001645405872736592 
- test set accuracy: 97.68%, test set average loss: 0.0019130350981839
- validation set accuracy: 98.36%, validation set average loss: 0.001457028130721301
- This is the best results obtained from my experiments.
- When Submitting https://github.com/RemonComputer/CatsVsDogsKaggle/tree/master/src/exp_1 at 4/ submission file I obtained a score of: 0.24735 which can make me in position 822 (but the leader board is closed)
- But the score obtained in this experiment is 0.08668 which can make me in position 299
- you can find the pre-trained model at: https://drive.google.com/open?id=1oBMEzHc_txPFKZF7P09LQmC63pKaa4Gt

## 1/
- This is a Transfer Learning exp.
- I added a new output layer and made it trainable from scratch.
- I made all the model parameters trainable.
- I trained the model for 10 iterations (27 min).
- train set accuracy: 99.42%, train set average loss: 0.0005929604744509561
- test set accuracy: 98.68%, test set average loss: 0.001211672317882767
- validation set accuracy: 98.68%, validation set average loss: 0.001052000113944814
- When I submitted the submission file, I obtained the score: 0.09105
- Although the loss and accuracies are better than the first exp, The final score isn't better, Which indicates some sort of overfitting.

## 2/
- This is a Transfer Learning exp.
- In this exp I borrowed the model from 0/ and fine tuned it (made all the model parameters trainable and trained it).
- I trained it for 10 epochs (22 min).
- We can say that this exp benfits from 0/ training epocs.
- train set accuracy: 99.585%, train set average loss: 0.0004015527827144069
- test set accuracy: 98.44%, test set average loss: 0.0013502538711181841
- validation set accuracy: 99.0%, validation set average loss: 0.000982945212131034
- When I submitted the submission file, I obtained the score: 0.10741
- Although the accuracy and the loss is much better than the previous two experiments but a clear overfitting is present which is indicated by the 99% accuracy and the small loss and this overfitting is evident by the obtained submission score.
- Although the higher accuracy is obtained by the validation set which is never seen by the model, but there might be some sort of overfitting.
- We might say that training and validating with a small number of training and validation data might blind us about future data so we are reaching the know conclusion which says that: more data is better, and it is not better for training only but for validation and testing also.
