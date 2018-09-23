import os
import ntpath
from itertools import cycle
import itertools
import random
import copy
from time import time
import cv2 as cv
from sklearn.metrics import accuracy_score
from skimage import io, transform
import matplotlib.pyplot as plt 
import numpy as np
import pandas
from pandas import DataFrame
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
from torch.autograd import Variable
import PIL

class CatsVsDogsDataset(Dataset):
    """Cats Vs Dogs Dataset from Kaggle ."""
    available_datasets = ['train','validation','test','submission']
    train_ratio = 0.8
    validation_ratio = 0.1
    test_ratio = 1 - (train_ratio + validation_ratio)
    train_folder_name = 'train'
    submission_folder_name = 'test'  

    def __init__(self, dataset_type, root_dir, transform=None):
        """
        Args:
            dataset_type (string): Type of the dataset you want to load.
            root_dir (string): Directory of the extracted dataset witch contains the train and test folder.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #self.root_dir = root_dir
        self.transform = transform
        if dataset_type == 'submission':
            images_subdirectory = self.submission_folder_name
        elif dataset_type in self.available_datasets:
            images_subdirectory = self.train_folder_name
        else:
            raise ValueError('dataset_type must be one of {}'.format(self.available_datasets))
        images_folder_path = os.path.join(root_dir, images_subdirectory)
        if dataset_type == 'submission':
            image_file_names = [image_file_name for image_file_name in os.listdir(images_folder_path)]
            self.images_paths = [os.path.join(images_folder_path, image_file_name) for image_file_name in image_file_names]
            self.labels = [int(image_file_name.replace('.jpg', '')) for image_file_name in image_file_names] #labels acts as ids in case of submission dataset
            #self.images = [io.imread(image_path) for image_path in images_paths]
            #if transform:
                #self.images = [transform(image) for image in self.images]
            #self.labels = [ -1 for image_path in range(self.images_paths)]
            print('Finished loading images')
            return
        dogs_images_paths = [os.path.join(images_folder_path, image_file_name) for image_file_name in os.listdir(images_folder_path) if 'dog' in image_file_name]
        cats_images_paths = [os.path.join(images_folder_path, image_file_name) for image_file_name in os.listdir(images_folder_path) if 'cat' in image_file_name]
        #assuming dogs and cats have equal image sizes
        if dataset_type == 'train':
            start_index = 0
            end_index = int(len(dogs_images_paths) * self.train_ratio)
        elif dataset_type == 'validation':
            start_index = int(len(dogs_images_paths) * self.train_ratio)
            end_index = int(len(dogs_images_paths) * (self.train_ratio + self.validation_ratio))
        elif dataset_type == 'test':
            start_index = int(len(dogs_images_paths) * (self.train_ratio + self.validation_ratio))
            end_index = len(dogs_images_paths)
        random.seed(7) #to make the same lists after shuffling every time the constructor called
        random.shuffle(dogs_images_paths)
        random.shuffle(cats_images_paths)
        dogs_image_paths_portion = dogs_images_paths[start_index: end_index]
        cats_image_paths_portion = cats_images_paths[start_index: end_index]
        print('dogs length: {}, cats length {}'.format(len(dogs_image_paths_portion), len(cats_image_paths_portion)))
        self.images_paths = dogs_image_paths_portion[:]
        self.images_paths.extend(cats_image_paths_portion)
        print('Final images_paths length: {}'.format(len(self.images_paths)))
        #print('Selected Paths: {}'.format(images_paths))
        #self.images = [io.imread(image_path) for image_path in images_paths]
        #if transform:
            #self.image = [transform(image) for image in self.images]
        self.labels = [1 for i in range(len(dogs_image_paths_portion))]
        self.labels.extend([0 for i in range(len(cats_image_paths_portion))])
        #print('Finished loading images')
            
    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image = io.imread(self.images_paths[idx])
        #print('Image Shape: {}'.format(image.shape))
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        #print('image tensor Size: {}'.format(image.size()))
        sample = (image, label) 
        return sample

class Rescale(object):
    """This is a Transformation that rescale the image in a sample to a given size, This transformation should be applied to the ndarray.
    Args:
        output_size (tuple or int): Desired output size.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else: 
            self.output_size = output_size

    def __call__(self, image):
        # h, w = image.shape[:2]
        # if isinstance(self.output_size, int):
        #     if h > w:
        #         new_h, new_w = self.output_size * h / w, self.output_size
        #     else:
        #         new_h, new_w = self.output_size, self.output_size * w / h
        # else:
        #     new_h, new_w = self.output_size
        # new_h, new_w = int(new_h), int(new_w)
        #img = transform.resize(image, (new_h, new_w))
        # img = cv.resize(image, (new_h, new_w), interpolation = cv.INTER_CUBIC)
        img = cv.resize(image, self.output_size, interpolation = cv.INTER_CUBIC)
        return img

class Normalize(object):
    """This is a transformation that normalizes the images."""
    def __call__(self, image):
        image = image.astype(np.float) / 256
        return image

class ToTensor(object):
    """This is a transformation that converts ndarrays in sample to Tensors."""
    def __call__(self, image):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)

def show_image(image, mean=None, std=None):
    ndim  = image.ndim
    if ndim == 3:
        if std is not None:
            for i in range(len(std)):
                image[:,:,i] *= std[i]
        if mean is not None:
            for i in range(len(mean)):
                image[:,:,i] += mean[i]
        plt.imshow(image)
    else:
        if std is not None:
            image *= std
        if mean is not None:
            image += mean
        plt.imshow(image, cmap='gray')
    plt.show()

def convert_image_from_tensor_and_display_it(tensor_image, mean=None, std=None):
    npimage =  tensor_image.numpy()
    ndim  = npimage.ndim
    #print('Resulted numpy image shape: {}'.format(npimage.shape))
    if npimage.shape[0] == 1:
        ndim -= 1
        npimage = npimage[0]
    if ndim == 3:    
        npimage = np.transpose(npimage, (1,2,0)) #use this line if you have multi-channel image ex: RGB or RGBA
        if std is not None:
            for i in range(len(std)):
                npimage[:,:,i] *= std[i]
        if mean is not None:
            for i in range(len(mean)):
                npimage[:,:,i] += mean[i]
        plt.imshow(npimage)
    else:
        if std is not None:
            npimage *= std
        if mean is not None:
            npimage += mean
        plt.imshow(npimage, cmap='gray')
    plt.show()

class Net(nn.Module):
    def __init__(self, model_file_path = 'model_state_dic.pytorch', reload_model = True, use_cuda_if_available = True):
        super(Net, self).__init__()
        '''
        #make the architect here
        self.layers = []
        #input_size = 1x256x256
        conv1 = nn.Conv2d(3, 3*16, kernel_size = 3)
        #conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.layers.append(conv1)
        #output_size = 16x254x254
        #max_pool1 = nn.MaxPool2d(kernel_size = 2)
        #self.layers.append(max_pool1)
        #output_size = 16x127x127
        #counter intutive network but it is a start
        conv2 = nn.Conv2d(3*16, 3*32, kernel_size = 4)
        #conv2 = nn.Conv2d(16, 32, kernel_size=4)
        self.layers.append(conv2)
        #output_size = 32x124x124
        #max_pool2 = nn.MaxPool2d(kernel_size = 2)
        #self.layers.append(max_pool2)
        #output_size = 32x62x62
        conv3 = nn.Conv2d(3*32, 3*64, kernel_size = 3)
        #conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.layers.append(conv3)
        #output_size = 64x60x60
        #max_pool3 = nn.MaxPool2d(kernel_size = 2)
        #self.layers.append(max_pool3)
        #output_size = 64x30x30
        conv4 = nn.Conv2d(3*64, 3*128, kernel_size = 3)
        #conv4 = nn.Conv2d(64, 128, kernel_size=3)
        self.layers.append(conv4)
        #output_size = 128x28x28
        #max_pool4 = nn.MaxPool2d(kernel_size = 2)
        #self.layers.append(max_pool4)
        #output_size = 128x14x14
        conv5 = nn.Conv2d(3*128, 3*256, kernel_size = 3)
        #conv5 = nn.Conv2d(128, 256, kernel_size=3)
        self.layers.append(conv5)
        #output_size = 256x12x12
        #max_pool5 = nn.MaxPool2d(kernel_size = 2)
        #self.layers.append(max_pool5)
        #output_size = 256x6x6
        conv6 = nn.Conv2d(3*256, 3*512, kernel_size = 3)
        #conv6 = nn.Conv2d(256, 512, kernel_size=3)
        self.layers.append(conv6)
        #output_size = 512x4x4
        #max_pool6 = nn.MaxPool2d(kernel_size = 2)
        #self.layers.append(max_pool6)
        #output_size = 512x2x2
        self.conv7 = nn.Conv2d(3*512, 3*1024, kernel_size = 2)
        #self.conv7 = nn.Conv2d(512, 1024, kernel_size=2)
        #self.layers.append(conv7)
        #output_size = 1024x1x1
        self.flat = nn.Linear(3*1024, 1)
        #self.flat = nn.Linear(1024, 1)
        self.pool = nn.MaxPool2d(kernel_size = 2)
        #self.output_function = nn.Sigmoid()
        #self.output_function = nn.ReLU()
        #self.output_function = nn.LeakyReLU()
        parameters = []
        for layer in self.layers:
            for parameter in layer.parameters():
                parameters.append(parameter)
        self.params = nn.ParameterList(parameters= parameters) #to make model.parameters() see the layer parameters so that it can optimize the layers
        '''
        # Here I will write the model
        self.pretrainedModel = models.resnet18(pretrained=True)
        #for param in self.pretrainedModel.parameters():
        #    param.requires_grad = False

        num_ftrs = self.pretrainedModel.fc.in_features
        # Parameters of newly constructed modules have requires_grad=True by default
        self.pretrainedModel.fc = nn.Linear(num_ftrs, 1)

        self.output_function = nn.Sigmoid()

        if reload_model == True and os.path.isfile(model_file_path):
            print('Loading model from: {}'.format(model_file_path))
            self.load_state_dict(torch.load(model_file_path))
            print('Model loaded successfully')
        else:
            print('Creating new model from Scratch')
        #these settings will override the state dictionary
        self.data_loader_workers = 4
        self.model_file_path = model_file_path
        self.device = torch.device('cuda' if use_cuda_if_available and torch.cuda.is_available() else 'cpu')
        print('Model is using: {}'.format(self.device))
        self = self.to(self.device) #transferring the model to the device
    
    def forward(self, input):
        '''
        for (i, layer) in enumerate(self.layers):
            #print('Classifying using layer {}, input size: {}'.format(i + 1, input.size()))
            input = self.pool(nn.functional.relu(layer(input)))
        #print('Classifying using layer {}, input size: {}'.format(len(self.layers) + 1, input.size()))
        input = nn.functional.relu(self.conv7(input))
        #print('Pre-Flat input size: {}'.format(input.size()))
        input = self.flat(input.view(-1, 3*1024))
        #input = self.flat(input.view(-1, 1024))
        output = self.output_function(input)
        #print('Network output size: {}'.format(output.size()))
        output = output.view(-1)
        return output
        '''
        #infer from the model here
        output = self.pretrainedModel(input)
        output = output.view(-1)
        output = self.output_function(output)
        return output
    
    def convert_probabilities_to_labels(self, probablility):
        labels = (probablility >= 0.5).long().view(-1)
        return labels

    def predict(self, input):
        probablility = self(input)
        labels = self.convert_probabilities_to_labels(probablility)
        return labels

    def myTrainining(self, optimizer, train_dataset, test_dataset, train_batch_size = 32, no_ephocs = 10, \
                test_accuracy_interval_in_epocs = 1, display_training_metrics_in_batches = 10):
        training_start_time = time()
        criterion = nn.BCELoss()
        #for name, param in model.named_parameters():
        #    if param.requires_grad:
        #        print('Parameter name: {}, data: {}'.format(name, param.data))
        #criterion = nn.MSELoss()
        #weights = []
        #sparsity_targets = []
        #for param in model.parameters():
        #    if param.requires_grad and param.data.dim() > 1: #I am excluding biases
        #        weights.append(param)
        #        sparsity_targets.append(Variable(torch.zeros_like(param.data)))
        #regularization_criteria = nn.L1Loss()
        #regularization_factor = 1e-1
        train_loader = DataLoader(train_dataset, batch_size = train_batch_size, shuffle = True, num_workers = self.data_loader_workers)
        test_loader = DataLoader(test_dataset, batch_size = train_batch_size, shuffle = True, num_workers = self.data_loader_workers)
        number_of_training_batches = len(train_loader)
        #test_iter = cycle(test_loader) # cyclic iterator
        #test_iter = iter(test_loader)
        best_model_loss = float('inf')
        best_model_state = None
        for epoch in range(no_ephocs):  # loop over the dataset multiple times
            # Validating
            # Validation is done  before Training To validate a newly uploaded model before corrupting it
            if epoch % test_accuracy_interval_in_epocs == 0:
                self.eval()
                with torch.no_grad():
                    total_test_examples = 0
                    correct_test_examples = 0
                    test_loss = 0
                    validation_start_time = time()
                    number_of_test_batches  = len(test_loader)
                    print('-------------------------------')
                    print('Validating...')
                    print('--------------')
                    for i, test_data in enumerate(test_loader, 0):
                        print('epoch: %d, test batch: %5d/%d' % (epoch + 1, i + 1, number_of_test_batches))
                        # get the inputs
                        test_inputs, test_labels = test_data
                        test_inputs = test_inputs.to(self.device)
                        test_labels = test_labels.to(self.device)
                        test_outputs = self(test_inputs)
                        test_loss += criterion(test_outputs, test_labels.float()).item()
                        # calculate the acumulated test accuracy
                        predicted_test_labels = self.convert_probabilities_to_labels(test_outputs)
                        current_correct_test_examples =  (predicted_test_labels == test_labels).sum().item()
                        correct_test_examples += current_correct_test_examples
                        total_test_examples += test_labels.size(0)
                    validation_end_time = time()
                    validation_loss = test_loss / total_test_examples
                    test_accuracy = correct_test_examples * 100 / total_test_examples
                    validation_time = validation_end_time - validation_start_time
                    print('--------------------------------')
                    print('Validation metrics:')
                    print('--------------------')
                    print('Validation loss: {}'.format(validation_loss))
                    print('Validation Accuracy: {}%'.format(test_accuracy))
                    print('Validation Took: {} seconds'.format(validation_time))
                    if best_model_loss > validation_loss:
                        best_model_loss = validation_loss
                        print('Best Model obtained so far .. Saving Model...')
                        state_dict = self.state_dict()
                        best_model_state = copy.deepcopy(state_dict)
                        torch.save(state_dict, self.model_file_path)
                        print('Model Saved')
                    print('---------------------------------')
             # Training
            self.train()
            epoch_start_time = time()
            running_loss = 0.0
            correct_training_example = 0.0
            total_training_examples = 0.0
            accumulated_batch_time = 0.0
            for i, data in enumerate(train_loader, 0):
                batch_start_time = time()
                print('epoch: %d, batch: %5d/%d' % (epoch + 1, i + 1, number_of_training_batches))
                # get the inputs
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self(inputs)
                #regularization_loss = 0
                #for (param, target) in zip(weights, sparsity_targets):
                    #regularization_loss += regularization_criteria(param, target)
                #print('output size: {}, label size: {}'.format(str(outputs.size()), str(labels.size())))
                loss = criterion(outputs, labels.float()) #+ regularization_factor * regularization_loss
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()
                # calculate the acumulated training accuracy
                predicted_labels = self.convert_probabilities_to_labels(outputs)
                current_corrected_examples =  (predicted_labels == labels).sum().item()
                correct_training_example += current_corrected_examples
                total_training_examples += labels.size(0)
                #print('Current outputs: {}'.format(outputs))
                #print('Predicted Labels: {}'.format(predicted_labels))
                #print('Correct Labels: {}'.format(labels))
                #print('Current Loss: {}'.format(loss.item()))
                #print('Current accuracy: {}'.format(current_corrected_examples * 100 / labels.size(0)))
                if i % display_training_metrics_in_batches == 0:
                    print('---------------------------------------------------------------------------------------------')
                    print('epoch: %d, batch: %5d/%d' % (epoch + 1, i + 1, number_of_training_batches))
                    print('---------------------------')
                    print('Training loss: %.3f' %
                        (running_loss / total_training_examples))
                    print('Training Accuracy: {}%'.format(correct_training_example * 100 / total_training_examples))
                    print('Average Batch Time: {} seconds'.format(accumulated_batch_time / display_training_metrics_in_batches))
                    running_loss = 0.0
                    correct_training_example = 0.0
                    total_training_examples = 0.0
                    accumulated_batch_time = 0.0
                    '''
                    with torch.no_grad():
                        #trying to predict the same training patch after updating the weights
                        outputs_again = self(inputs)
                        predicted_labels_again = self.convert_probabilities_to_labels(outputs_again)
                        correct_training_examples_again= (predicted_labels_again == labels).sum().item()
                        total_training_examples_again = labels.size(0)
                        print('Training accuracy of the last patch after updating the gradient: {}%'.format(correct_training_examples_again * 100 / total_training_examples_again))
                        #try:
                        #    (test_inputs, test_labels) = next(test_iter)
                        #except StopIteration: #to simulate indefinite cycling through test set
                        #    test_iter = iter(test_loader)
                        #    (test_inputs, test_labels) = next(test_iter)
                        test_loss = 0.0
                        correct_test_labels = 0
                        total_test_labels = 0
                        for _ in range(no_test_batches):
                            (test_inputs, test_labels) = next(test_iter)
                            test_inputs = test_inputs.to(self.device)
                            test_labels = test_labels.to(self.device)
                            test_outputs = self(test_inputs)
                            test_loss += criterion(test_outputs, test_labels.float())
                            predicted_test_labels = self.convert_probabilities_to_labels(test_outputs)
                            correct_test_labels += (test_labels == predicted_test_labels).sum().item()
                            total_test_labels += test_labels.size(0)
                        test_accuracy = correct_test_labels * 100 / total_test_labels
                        test_loss *=  train_batch_size / (test_batch_size  * no_test_batches)
                        print('Test loss: {:.3f}'.format(test_loss))
                        print('Test accuracy: {}%'.format(test_accuracy))
                        if test_accuracy > 98.0:
                            print('Quiting training because Test accuracy has passed 98%')
                            print('This is a condition that is made after seeing the network vibrations')
                            return
                    print('--------------------------------------------------------------------------------------------------------------------')
                '''
                batch_end_time = time()
                accumulated_batch_time += (batch_end_time - batch_start_time)
                #print('Batch took: {} seconds'.format(batch_end_time - batch_start_time))
            #Zeroeing parameters so that values don't pass from epoch to another 
            running_loss = 0.0
            correct_training_example = 0.0
            total_training_examples = 0.0
            accumulated_batch_time = 0.0
            '''
            if epoch % model_save_intervals_in_ephocs == 0:
                print('Saving model at {} at the end of epoch: {} .....'.format(self.model_file_path, epoch + 1))
                state_dict = self.state_dict()
                torch.save(state_dict, self.model_file_path)
                print('Model Saved')
                print('-----------------------------------------------------------------------------------------------------------')
            '''
            epoch_end_time = time()
            print('Epoch {} took: {} seconds'.format(epoch + 1, epoch_end_time - epoch_start_time))
            print('----------------------------------------------------------------------------------------------------------------------')
        #print('----------------------------------------------------------------------------------------------------------------------------')
        #print('Saving model at the end of training')
        #state_dict = self.state_dict()
        #torch.save(state_dict, self.model_file_path)
        #print('Model Saved')
        #if no_ephocs % test_accuracy_interval_in_epocs != 0:
        # Validate the model at the end of training
        self.eval()
        with torch.no_grad():
            total_test_examples = 0
            correct_test_examples = 0
            test_loss = 0
            validation_start_time = time()
            number_of_test_batches  = len(test_loader)
            print('-------------------------------')
            print('Validating...')
            print('--------------')
            for i, test_data in enumerate(test_loader, 0):
                print('epoch: %d, test batch: %5d/%d' % (epoch + 1, i + 1, number_of_test_batches))
                # get the inputs
                test_inputs, test_labels = test_data
                test_inputs = test_inputs.to(self.device)
                test_labels = test_labels.to(self.device)
                test_outputs = self(test_inputs)
                test_loss += criterion(test_outputs, test_labels.float()).item()
                # calculate the acumulated test accuracy
                predicted_test_labels = self.convert_probabilities_to_labels(test_outputs)
                current_correct_test_examples =  (predicted_test_labels == test_labels).sum().item()
                correct_test_examples += current_correct_test_examples
                total_test_examples += test_labels.size(0)
            validation_end_time = time()
            validation_loss = test_loss / total_test_examples
            test_accuracy = correct_test_examples * 100 / total_test_examples
            validation_time = validation_end_time - validation_start_time
            print('--------------------------------')
            print('Validation metrics:')
            print('--------------------')
            print('Validation loss: {}'.format(validation_loss))
            print('Validation Accuracy: {}%'.format(test_accuracy))
            print('Validation Took: {} seconds'.format(validation_time))
            if best_model_loss > validation_loss:
                best_model_loss = validation_loss
                print('Best Model obtained so far .. Saving Model...')
                state_dict = self.state_dict()
                best_model_state = copy.deepcopy(state_dict)
                torch.save(state_dict, self.model_file_path)
                print('Model Saved')
            print('---------------------------------')
        if best_model_state:
            self.load_state_dict(best_model_state)
        print('-----------------------------------------------------------------------------------------------------------')
        print('Training Finished')
        training_end_time = time()
        print('Training for {} epochs took: {} seconds'.format(no_ephocs, training_end_time - training_start_time))
        

    def test(self, test_dataset, dataset_name = 'test', batch_size = 32):
        test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = self.data_loader_workers)
        #predicted_labels = []
        #true_labels = []
        number_of_batches = len(test_loader)
        number_of_samples = len(test_dataset)
        number_of_correct_samples = 0
        print()
        print('Testing dataset {}'.format(dataset_name))
        print('-----------------------------------------')
        criterion = nn.BCELoss()
        loss = 0.0
        self.eval()
        with torch.no_grad():
            # enumerate on loader
            for (i, (images, labels)) in enumerate(test_loader):
                print('Processing batch {}/{} . . .'.format(i + 1, number_of_batches))
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self(images)
                #batch_labels = self.predict(images)
                batch_labels = self.convert_probabilities_to_labels(outputs)
                number_of_correct_samples_in_batch = (batch_labels == labels).sum().item()
                number_of_correct_samples += number_of_correct_samples_in_batch
                loss += criterion(outputs, labels.float()).item()
                #true_labels.extend(labels)
                # extend output list with outputs.numpy
                #predicted_labels.extend(batch_labels)
                # end enumeration
            #accuracy = accuracy_score(true_labels, predicted_labels) * 100
            accuracy = number_of_correct_samples * 100 / number_of_samples
            average_loss = loss / number_of_samples
            print('{} set accuracy: {}%'.format(dataset_name, accuracy))
            print('{} set average loss: {}'.format(dataset_name, average_loss))
            print('----------------------------------------------------------------------------')
            print()
           

    def generate_submission_file(self, submission_dataset, submission_file_path, batch_size = 32):
        # create dataset_loader with batch size
        submission_loader = DataLoader(submission_dataset, batch_size = batch_size, shuffle = False, num_workers = self.data_loader_workers)
        # create empty id list, empty output list
        ids =[]
        outputs = []
        number_of_batches = len(submission_loader)
        print()
        print('Generating Submission file:')
        print('-----------------------------')
        self.eval()
        with torch.no_grad():
            # enumerate on loader
            for (i, (images, labels)) in enumerate(submission_loader):
                print('Processing batch {}/{} . . .'.format(i + 1, number_of_batches))
                images = images.to(self.device)
                labels = labels.to(self.device)
                # call self(batch) to return outputs
                batch_outputs = self(images)
                # extend the id list with the labels.numpy() or as a slow iterate on labels and add them to id
                ids.extend(labels.cpu().numpy())
                # extend output list with outputs.numpy
                outputs.extend(batch_outputs.cpu().numpy())
                # end enumeration 
        # create dataframe that holds the {'id': id_list, 'label': output_list}
        submission_dataframe = DataFrame({'id':ids, 'label':outputs})
        # dataframe.to_csv('submission_file' , index=False) 
        submission_dataframe.to_csv(submission_file_path, index=False)
        print('Submission file generated at: {}'.format(submission_file_path))

    def print_model_parameters(self):
        print()
        print('Printing model Parameters:')
        print('---------------------------')
        for param in self.parameters():
            if param.requires_grad:
                print(param)
        print('--------------------------------------------------------------------------------------------')

if __name__ == '__main__':
    torch.manual_seed(7)
    print('is cuda available: {}'.format(torch.cuda.is_available()))
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print('you are using the {}'.format(device))
    #, transforms.Grayscale()
    train_transform = transforms.Compose([transforms.ToPILImage(),
                                          #transforms.Grayscale(),
                                          transforms.Resize((224, 224), interpolation = PIL.Image.BILINEAR),
                                          transforms.ColorJitter(),
                                          transforms.RandomAffine(degrees = 15, translate=(0.1, 0.1), scale=(0.5, 2), resample = PIL.Image.NEAREST),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    test_transform = transforms.Compose([transforms.ToPILImage(),
                                         #transforms.Grayscale(),
                                         transforms.Resize((224, 224), interpolation=PIL.Image.BILINEAR),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    augmented_train_dataset = CatsVsDogsDataset('train', '../../../Dataset/', transform=train_transform)
    '''
    future_transform = current_transform
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=1,
                                          shuffle=True, num_workers=2)
    dataiter = iter(trainloader)
    class_labels = ['cat', 'dog']
    for i in range(5):
        (imgs, labels) = dataiter.next()
        print('Current image: {}'.format(class_labels[labels[0]]))
        #print('image type: {}'.format(type(imgs[0])))
        convert_image_from_tensor_and_display_it(imgs[0], 0.5, 0.5) #, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    '''
    #--------------------------------------------------------------------------------------------------------------
    model = Net(model_file_path = 'model_state_dict.pytorch', reload_model=True, use_cuda_if_available=True)
    test_dataset = CatsVsDogsDataset('test', '../../../Dataset/', transform=test_transform)
    validation_dataset = CatsVsDogsDataset('validation', '../../../Dataset/', transform=test_transform)
    model_parameters = model.parameters()
    #optimizer =  optim.SGD(model_parameters, lr=0.01, momentum=0.9)
    optimizer = optim.Adam(model_parameters, lr=0.0001)
    #optimizer, train_dataset, test_dataset, train_batch_size = 32, no_ephocs = 10, \
    #            test_accuracy_interval_in_epocs = 1, display_training_metrics_in_batches = 10
    model.myTrainining(optimizer, augmented_train_dataset, test_dataset, train_batch_size = 32, no_ephocs = 10, \
                       test_accuracy_interval_in_epocs = 1, display_training_metrics_in_batches = 10)
    #model.print_model_parameters()
    original_train_dataset = CatsVsDogsDataset('train', '../../../Dataset/', transform=test_transform)
    model.test(original_train_dataset, 'train', batch_size = 32)
    model.test(test_dataset, 'test', batch_size = 32)
    model.test(validation_dataset, 'validation', batch_size = 32)
    submission_dataset = CatsVsDogsDataset('submission', '../../../Dataset/', transform=test_transform)
    model.generate_submission_file(submission_dataset, 'submission.csv', batch_size = 32)
