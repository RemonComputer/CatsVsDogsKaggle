import os
import ntpath
from itertools import cycle
import cv2 as cv
from sklearn.metrics import accuracy_score
from skimage import io, transform
import matplotlib.pyplot as plt 
import numpy as np
import pandas
from pandas import DataFrame
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
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
            self.labels = [image_file_name.replace('.jpg', '') for image_file_name in image_file_names] #labels acts as ids in case of submission dataset
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
    print('Resulted numpy image shape: {}'.format(npimage.shape))
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
        #make the architect here
        self.layers = []
        #input_size = 1x256x256
        conv1 = nn.Conv2d(1, 64, kernal_size = 3)
        self.layers.append(conv1)
        #output_size = 64x254x254
        #max_pool1 = nn.MaxPool2d(kernal_size = 2)
        #self.layers.append(max_pool1)
        #output_size = 64x127x127
        #counter intutive network but it is a start
        conv2 = nn.Conv2d(64, 32, kernal_size = 4)
        self.layers.append(conv2)
        #output_size = 32x124x124
        #max_pool2 = nn.MaxPool2d(kernal_size = 2)
        #self.layers.append(max_pool2)
        #output_size = 32x62x62
        conv3 = nn.Conv2d(32, 16, kernal_size = 3)
        self.layers.append(conv3)
        #output_size = 16x60x60
        #max_pool3 = nn.MaxPool2d(kernal_size = 2)
        #self.layers.append(max_pool3)
        #output_size = 16x30x30
        conv4 = nn.Conv2d(16, 8, kernal_size = 3)
        self.layers.append(conv4)
        #output_size = 8x28x28
        #max_pool4 = nn.MaxPool2d(kernal_size = 2)
        #self.layers.append(max_pool4)
        #output_size = 8x14x14
        conv5 = nn.Conv2d(8, 4, kernal_size = 3)
        self.layers.append(conv5)
        #output_size = 4x12x12
        #max_pool5 = nn.MaxPool2d(kernal_size = 2)
        #self.layers.append(max_pool5)
        #output_size = 4x6x6
        conv6 = nn.Conv2d(4, 2, kernal_size = 3)
        self.layers.append(conv6)
        #output_size = 2x4x4
        #max_pool6 = nn.MaxPool2d(kernal_size = 2)
        #self.layers.append(max_pool6)
        #output_size = 2x2x2
        conv7 = nn.Conv2d(2, 1, kernal_size = 2)
        self.layers.append(conv7)
        self.pool = nn.MaxPool2d(kernal_size = 2)
        self.sigmoid = nn.Sigmoid()

        if reload_model == True and os.path.isfile(model_file_path):
            print('Loading model from: {}'.format(model_file_path))
            self.load_state_dict(torch.load(model_file_path))
            print('Model loaded successfully')
        else:
            print('Creating new model from Scratch')
        #these settings will override the state dictionary
        self.data_loader_workers = 2
        self.model_file_path = model_file_path
        self.device = torch.device('cuda' if use_cuda_if_available and torch.cuda.is_available() else 'cpu')
        self = self.to(device) #transferring the model to the device
    
    def forward(self, input):
        for layer in self.layers:
            input = self.pool(nn.functional.relu(layer(input)))
        output = self.sigmoid(input)
        return output
    
    def convert_probabilities_to_labels(self, probablility):
        labels = (probablility >= 0.5).long()
        return labels

    def predict(self, input):
        probablility = self(input)
        labels = self.convert_probabilities_to_labels(probablility)
        return labels

    def train(self, optimizer, train_dataset, test_dataset, train_batch_size = 32, test_batch_size = 128, no_ephocs = 10, \
                model_save_intervals_in_ephocs = 5, test_accuracy_interval_in_batches = 100):
        criterion = nn.BCELoss()
        train_loader = DataLoader(train_dataset, batch_size = train_batch_size, shuffle = True, num_workers = self.data_load_workers)
        test_loader = DataLoader(test_dataset, batch_size = test_batch_size, shuffle = False, num_workers = self.data_load_workers)
        number_of_training_batches = len(train_loader)
        test_iter = cycle(test_loader) # cyclic iterator
        for epoch in range(no_ephocs):  # loop over the dataset multiple times
            running_loss = 0.0
            correct_training_example = 0.0
            total_training_examples = 0.0
            for i, data in enumerate(train_loader, 0):
                # get the inputs
                inputs, labels = data
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()
                # calculate the acumulated training accuracy
                predicted_labels = self.convert_probabilities_to_labels(outputs)
                correct_training_example += (predicted_labels == labels).sum().item()
                total_training_examples += labels.size(0)
                if i % test_accuracy_interval_in_batches == 0 and i != 0:
                    print('epoch: %d, batch: %5d' % (epoch + 1, i + 1))
                    print('---------------------------')
                    print('Training loss: %.3f' %
                        (running_loss / test_accuracy_interval_in_batches))
                    running_loss = 0.0
                    print('Training Accuracy: {}%%'.format(correct_training_example * 100 / total_training_examples))
                    correct_training_example = 0.0
                    total_training_examples = 0.0
                    with torch.no_grad():
                        (test_inputs, test_labels) = test_iter.next()
                        test_outputs = self(test_inputs)
                        test_loss = criterion(test_outputs, test_labels) * train_batch_size / test_batch_size 
                        print('Test loss: {:.3f}'.format())
                        predicted_test_labels = self.convert_probabilities_to_labels(outputs)
                        correct_test_labels = (test_labels == predicted_test_labels).sum().item()
                        test_accuracy = correct_test_labels * 100 / test_labels.size(0)
                        print('Test accuracy: {}%%'.format(test_accuracy))
                    print('--------------------------------------------------------------------------------------------------------------------')
            if epoch % model_save_intervals_in_ephocs == 0 and epoch != 0:
                print('Saving model at {} at the end of epoch: {} .....'.format(self.model_file_path, epoch + 1))
                state_dict = self.state_dict()
                torch.save(state_dict, self.model_file_path)
                print('Model Saved')
                print('-----------------------------------------------------------------------------------------------------------')
        print('Finished Training')
        

    def test(self, test_dataset, dataset_name = 'test', batch_size = 32):
        test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = self.data_load_workers)
        predicted_labels = []
        true_labels = []
        number_of_batches = len(test_loader)
         with torch.no_grad():
            # enumerate on loader
            for (i, (images, labels)) in enumerate(submission_loader):
            	print('Processing batch {}/{} . . .'.format(i + 1, number_of_batches))
                batch_labels = self.predict(images)
                true_labels.extend(labels)
                # extend output list with outputs.numpy
                predicted_labels.extend(batch_outputs)
                # end enumeration
           accuracy = accuracy_score(true_labels, predicted_labels) * 100
           print('{} set accuracy: {}%%'.format(dataset_name, accuracy))
           

    def generate_submission_file(self, submission_dataset, submission_file_path, batch_size = 32):
        # create dataset_loader with batch size
        submission_loader = DataLoader(submission_dataset, batch_size = batch_size, shuffle = False, num_workers = self.data_load_workers)
        # create empty id list, empty output list
        ids =[]
        outputs = []
        number_of_batches = len(submission_loader)
        with torch.no_grad():
            # enumerate on loader
            for (i, (images, labels)) in enumerate(submission_loader):
            	print('Processing batch {}/{} . . .'.format(i + 1, number_of_batches))
                # call self(batch) to return outputs
                batch_outputs = self(images)
                # extend the id list with the labels.numpy() or as a slow iterate on labels and add them to id
                ids.extend(labels)
                # extend output list with outputs.numpy
                outputs.extend(batch_outputs)
                # end enumeration 
        # create dataframe that holds the {'id': id_list, 'label': output_list}
        submission_dataframe = DataFrame({'id':ids, 'label':outputs})
        # dataframe.to_csv('submission_file' , index=False) 
        submission_dataframe.to_csv(submission_file_path, index=False)

if __name__ == '__main__':
    torch.manual_seed(7)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('you are using the {}'.format(device))
    current_transform = transforms.Compose([transforms.ToPILImage(), transforms.Grayscale(), transforms.Resize((256, 256), interpolation=PIL.Image.BILINEAR),transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]) #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    train_dataset = CatsVsDogsDataset('train', '../../../Dataset/', transform=current_transform) 
    future_transform = current_transform
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=1,
                                          shuffle=True, num_workers=2)
    dataiter = iter(trainloader)
    class_labels = ['cat', 'dog']
    for i in range(5):
        (imgs, labels) = dataiter.next()
        print('Current image: {}'.format(class_labels[labels[0]]))
        print('image type: {}'.format(type(imgs[0])))
        convert_image_from_tensor_and_display_it(imgs[0], 0.5, 0.5) #, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    #--------------------------------------------------------------------------------------------------------------
    model = Net(model_file_path = 'model_state_dict.pytorch', reload_model=False, use_cuda_if_available=True)
    train_dataset = CatsVsDogsDataset('test', '../../../Dataset/', transform=current_transform)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train(optimizer, train_dataset, test_dataset, train_batch_size = 32, test_batch_size = 128, no_ephocs = 10, model_save_intervals_in_ephocs = 2, test_accuracy_interval_in_batches = 100)
    model.test(train_dataset, 'train', batch_size = 32)
    model.test(test_dataset, 'test', batch_size = 32)
    submission_dataset = CatsVsDogsDataset('submission', '../../../Dataset/', transform=current_transform)
    model.generate_submission_file(submission_dataset, 'submission.csv')