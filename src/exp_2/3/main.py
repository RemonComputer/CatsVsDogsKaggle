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
    def __init__(self, model_file_path = 'model_state_dic.pth', reload_model = True, use_cuda_if_available = True):
        super(Net, self).__init__()
        self.pretrainedModel = models.resnet152(pretrained=True)
        for param in self.pretrainedModel.parameters():
            param.requires_grad = False

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
        criterion = nn.BCELoss(size_average=False)
        train_loader = DataLoader(train_dataset, batch_size = train_batch_size, shuffle = True, num_workers = self.data_loader_workers)
        test_loader = DataLoader(test_dataset, batch_size = train_batch_size, shuffle = True, num_workers = self.data_loader_workers)
        number_of_training_batches = len(train_loader)
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
                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()
                # calculate the acumulated training accuracy
                predicted_labels = self.convert_probabilities_to_labels(outputs)
                current_corrected_examples =  (predicted_labels == labels).sum().item()
                correct_training_example += current_corrected_examples
                total_training_examples += labels.size(0)
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
                batch_end_time = time()
                accumulated_batch_time += (batch_end_time - batch_start_time)
            #Zeroeing parameters so that values don't pass from epoch to another 
            running_loss = 0.0
            correct_training_example = 0.0
            total_training_examples = 0.0
            accumulated_batch_time = 0.0
            epoch_end_time = time()
            print('Epoch {} took: {} seconds'.format(epoch + 1, epoch_end_time - epoch_start_time))
            print('----------------------------------------------------------------------------------------------------------------------')
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
        number_of_batches = len(test_loader)
        number_of_samples = len(test_dataset)
        number_of_correct_samples = 0
        print()
        print('Testing dataset {}'.format(dataset_name))
        print('-----------------------------------------')
        criterion = nn.BCELoss(size_average=False)
        loss = 0.0
        self.eval()
        with torch.no_grad():
            # enumerate on loader
            for (i, (images, labels)) in enumerate(test_loader):
                print('Processing batch {}/{} . . .'.format(i + 1, number_of_batches))
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self(images)
                batch_labels = self.convert_probabilities_to_labels(outputs)
                number_of_correct_samples_in_batch = (batch_labels == labels).sum().item()
                number_of_correct_samples += number_of_correct_samples_in_batch
                loss += criterion(outputs, labels.float()).item()
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
        # scaled_outputs = []
        # scale_factor = 10
        # for out in outputs:
        #     if out >= 0.5:
        #         scaled_out = 1 - (1 - out) / scale_factor
        #     else:
        #         scaled_out = out / scale_factor
        #     scaled_outputs.append(scaled_out)
        # outputs = scaled_outputs
        submission_dataframe = DataFrame({'id':ids, 'label':outputs})
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
    train_transform = transforms.Compose([transforms.ToPILImage(),
                                          transforms.RandomResizedCrop(224),
                                          #transforms.Resize((224, 224), interpolation = PIL.Image.BILINEAR),
                                          transforms.ColorJitter(),
                                          #transforms.RandomAffine(degrees = 15, translate=(0.1, 0.1), scale=(0.5, 2), resample = PIL.Image.NEAREST),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    test_transform = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize((224, 224), interpolation=PIL.Image.BILINEAR),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    augmented_train_dataset = CatsVsDogsDataset('train', '../../../Dataset/', transform=train_transform)
    #--------------------------------------------------------------------------------------------------------------
    model = Net(model_file_path = 'model_state_dict.pth', reload_model=True, use_cuda_if_available=True)
    test_dataset = CatsVsDogsDataset('test', '../../../Dataset/', transform=test_transform)
    validation_dataset = CatsVsDogsDataset('validation', '../../../Dataset/', transform=test_transform)
    model_parameters = model.pretrainedModel.fc.parameters()
    #optimizer =  optim.SGD(model_parameters, lr=0.01, momentum=0.9)
    optimizer = optim.Adam(model_parameters, lr=0.0001)
    model.myTrainining(optimizer, augmented_train_dataset, test_dataset, train_batch_size = 32, no_ephocs = 10, \
                       test_accuracy_interval_in_epocs = 1, display_training_metrics_in_batches = 100)
    #model.print_model_parameters()
    original_train_dataset = CatsVsDogsDataset('train', '../../../Dataset/', transform=test_transform)
    model.test(original_train_dataset, 'train', batch_size = 32)
    model.test(test_dataset, 'test', batch_size = 32)
    model.test(validation_dataset, 'validation', batch_size = 32)
    submission_dataset = CatsVsDogsDataset('submission', '../../../Dataset/', transform=test_transform)
    model.generate_submission_file(submission_dataset, 'submission.csv', batch_size = 32)
