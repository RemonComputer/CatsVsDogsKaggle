import cv2 as cv
from skimage import io, transform
import matplotlib.pyplot as plt 
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import PIL

class CatsVsDogsDataset(Dataset):
    """Cats Vs Dogs Dataset from Kaggle ."""
    available_datasets = ['train','crossvalidation','test','submission']
    train_ratio = 0.8
    cross_validation_ratio = 0.1
    test_ratio = 1 - (train_ratio + cross_validation_ratio)
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
            self.images_paths = [os.path.join(images_folder_path, image_file_name) for image_file_name in os.listdir(images_folder_path)]
            #self.images = [io.imread(image_path) for image_path in images_paths]
            #if transform:
                #self.images = [transform(image) for image in self.images]
            self.labels = [ -1 for i in range(len(self.images_paths))]
            print('Finished loading images')
            return
        dogs_images_paths = [os.path.join(images_folder_path, image_file_name) for image_file_name in os.listdir(images_folder_path) if 'dog' in image_file_name]
        cats_images_paths = [os.path.join(images_folder_path, image_file_name) for image_file_name in os.listdir(images_folder_path) if 'cat' in image_file_name]
        #assuming dogs and cats have equal image sizes
        if dataset_type == 'train':
            start_index = 0
            end_index = int(len(dogs_images_paths) * self.train_ratio)
        elif dataset_type == 'crossvalidation':
            start_index = int(len(dogs_images_paths) * self.train_ratio)
            end_index = int(len(dogs_images_paths) * (self.train_ratio + self.cross_validation_ratio))
        elif dataset_type == 'test':
            start_index = int(len(dogs_images_paths) * (self.train_ratio + self.cross_validation_ratio))
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

if __name__ == '__main__':
    torch.manual_seed(7)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('you are using the {}'.format(device))
    current_transform = transforms.Compose([transforms.ToPILImage(), transforms.Grayscale(), transforms.Resize((256, 256), interpolation=PIL.Image.BILINEAR),transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]) #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    dataset = CatsVsDogsDataset('train', '../../../Dataset/', transform=current_transform) 
    future_transform = current_transform
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                          shuffle=True, num_workers=2)
    dataiter = iter(trainloader)
    class_labels = ['cat', 'dog']
    for i in range(5):
        (imgs, labels) = dataiter.next()
        print('Current image: {}'.format(class_labels[labels[0]]))
        print('image type: {}'.format(type(imgs[0])))
        convert_image_from_tensor_and_display_it(imgs[0], 0.5, 0.5) #, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
