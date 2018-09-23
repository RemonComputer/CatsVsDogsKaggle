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
