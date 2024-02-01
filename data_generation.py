import numpy as np
import struct
from array import array
from os.path  import join
import random
import matplotlib.pyplot as plt
import cv2
import pickle
from tqdm import tqdm
import os
import torch

#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return np.asarray(images), np.asarray(labels)
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)  


input_path = './mnist/'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

#
# Helper function to show a list of images with their relating titles
#
def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30,20))
    index = 1    
    for x in zip(images, title_texts):        
        image = x[0]        
        title_text = x[1]
        plt.subplot(rows, cols, index)        
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 15);        
        index += 1

#
# Load MINST dataset
#
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

def _alter_pixel(pixel, prob):
    if np.random.random() < prob:
        return np.random.randint(0, 256, dtype=np.uint8)
    return pixel

def _alter(images, prob):
    return np.vectorize(_alter_pixel)(images, prob)
    

def _make_video(image, prob, length):
    return np.array([_alter(image, prob) for _ in range(length)]).astype(np.uint8)

def save_video(video, fps=10, filepath='output.mp4'):
    size = 28, 28
    out = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
    for _ in range(video.shape[0]):
        out.write(video[_, :, :])
    out.release()

# returns a tuple of (data, labels), where data is the video data and labels is the label for each frame
def _make_one(images, labels, 
             img_len_func = lambda: np.random.geometric(0.1),
             no_img_func = lambda: np.random.geometric(0.1),
             noise_prob = 0.1):
    data = np.empty((0, 28, 28), dtype=np.uint8)
    out_labels = np.empty((0), dtype=np.uint8)
    for _ in range(no_img_func()):
        length = img_len_func()
        idx = random.randint(0, images.shape[0]-1)
        vid = _make_video(images[idx], noise_prob, length)
        data = np.concatenate((data, vid), axis=0)
        out_labels = np.concatenate((out_labels, np.full((length), labels[idx])), axis=0) 
    return np.asarray(data), np.asarray(out_labels)


def gen_data(num_of_cases, noise_prob = 0.1, no_imgs_func = lambda: np.random.geometric(0.1), img_len_func = lambda: np.random.geometric(0.1)):
    videos = []
    labels = []
    
    for i in tqdm(range(num_of_cases), desc="Processing", unit="iteration"):
        video, label = _make_one(x_train, y_train, img_len_func, no_imgs_func, noise_prob)
        videos.append(video)
        labels.append(label)
        
    return videos, labels


def gen_and_export_data(num_of_cases, noise_prob = 0.1, no_imgs_func = lambda: np.random.geometric(0.1), img_len_func = lambda: np.random.geometric(0.1),
                        data_path = './generated/data.pickle', labels_path = './generated/labels.pickle'):
    videos, labels = gen_data(num_of_cases, noise_prob, no_imgs_func, img_len_func)

    with open(data_path, 'wb') as file:
        pickle.dump(videos, file)

    with open(labels_path, 'wb') as file:
        pickle.dump(labels, file)
        
    print(f"saved {num_of_cases} videos to {data_path} and {labels_path}")
    data_size = os.path.getsize(data_path)
    labels_size = os.path.getsize(labels_path)

    print(f"Size of data file: {data_size//1024**2} MB")
    print(f"Size of labels file: {labels_size//1024**2} MB")


def get_data_squeezed(data_path = './generated/data.pickle', labels_path = './generated/labels.pickle'):
    with open(data_path, 'rb') as file:
        data = [torch.tensor(x
                             .reshape(x.shape[0], -1)
                             .astype(np.float32)) for x in pickle.load(file)]

    with open(labels_path, 'rb') as file:
        labels = [torch.tensor(x.astype(np.float32)).long() for x in pickle.load(file)]
    return data, labels

def get_data_unsqueezed(data_path = './generated/data.pickle', labels_path = './generated/labels.pickle'):
    with open(data_path, 'rb') as file:
        data = [torch.tensor(x
                             .reshape(x.shape[0], 1, 28, 28)
                             .astype(np.float32)) for x in pickle.load(file)]

    with open(labels_path, 'rb') as file:
        labels = [torch.tensor(x.astype(np.float32)).long() for x in pickle.load(file)]
    return data, labels



