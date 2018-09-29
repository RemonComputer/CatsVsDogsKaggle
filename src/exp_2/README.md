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
- You can find the pre-trained mode at: https://drive.google.com/open?id=1QSLouMdrs49mxIityo47G1zBMc65n2vf

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
- You can find the pre-trained model at: https://drive.google.com/open?id=1EaypuujZWikOcymzF_Znn2EuC7qn_nE2

## 3/
- This is a Transfer Learning exp.- This is a Transfer Learning exp.- This is a Transfer Learning exp.
- In this exp, I followed the same conclusion from the previous experimensts (using pre-trained models as feature extractors is better than fine tunning them - from the point of view of the submission score).
- I used ResNet152 because it is the most accurate model in pytorch pre-trained models (in imagenet classification accuracy).
### First Sub-experiment:
- I trained the model with the previous transformaions (Data augmentation transformation).
- I trained it for 10 epochs.
- Training took 50 min.
- Train set accuracy: 98.76%, train set average loss: 0.001304504435567651
- Test set accuracy: 98.32%, Test set average loss: 0.0015543955907225608
- Validation set accuracy: 99.24%, Validation set average loss: 0.0011028685204684733
- Submission Score: 0.06459
- Submission position: 142 of 1314
### Second Sub-experiment:
- Removing the Affine Transformation from the Training data augmented transformations, Replacing it by RandomResizedCrop Transformation.
- I trained it for 10 epochs.
- Train set accuracy: 98.92%, Train set average loss: 0.001024433937104186
- Test set accuracy: 98.88%, Test set average loss: 0.0011318141617812215
- Validation set accuracy: 99.32%, Validation set average loss: 0.0008427965111099183
- Submission score: 0.05795
- Submission position: 105 of 1314
- It is clear from the previous two sub-experiments that using RandomResizedCrop is better than affine and we can understand that the the test, validation and submission photos isn't subject to Affine Transformations, So we can deduce that affine Transformations is a bit unrealistic.
### Third Sub-experiment:
- This is the same experiment as the previous sub-experiment but I took the model and trained it for additional 72 iterations.
- Train set accuracy: 99.185%, train set average loss: 0.0007505370588231017
- Test set accuracy: 99.04%, Test set average loss: 0.000939809382148087
- Validation set accuracy: 99.44%, Validation set average loss: 0.000629462133301422
- Submission score: 0.05981
- Submission position: 119
- Despite the fact that this model is more trained and has a better accuracy (Train, Test, Validation) and better validation score (Train, Test, Validation), but we still have the problem that preformance on the train, validation and test datasets doesn't reflect on the submission score.
### Fourth Sub-experiment:
- This is the same as the previous sub-experiment.
- I Took the model from the previous sub-experiment.
- I didn't train a new model.
- I only modified the function that outputs the score to decrease it by a factor of 10
- Submission score: 0.07480
- Submission position: 216
- despite the fact that the accuracy is about 99% so it is intutive to say that decreasing the score most of the time will make a better total score, but the score function is not linear and didn't work that way.
### Fifth Sub-experiment:
- I fixed a bug in calculating the loss (loss was the average of the mean, I corrected it to be the average of the sum), This bug only changes the loss by a constant factor so it is not critical to the model training or evaluation.
- The model of the Second Sub-experiment has been overridden so I had to redo it. 
- Training took 49 mins.
- Train set accuracy: 98.98%, Train set average loss: 0.03300796768404543
- Test set accuracy: 98.92%, Test set average loss: 0.03538055700659752
- Validation set accuracy: 99.28%, Validation set average loss: 0.026985937616974116
- Score: 0.05779
- Position: 104
- You can find the pre-trained model at: https://drive.google.com/open?id=1eiKCdOkDl75je1VCwP9qac-a5C1WJZa4
- But a question remains why score on validation and test sets doesn't reflect correctly on the submission score?, also why the Submission score is consideably higher that the validation and test scores? 