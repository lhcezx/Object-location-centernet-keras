import numpy as np
import tensorflow as tf
import cv2

from utils import generate_center_heatmap
from matplotlib import pyplot as plt
import pandas as pd
import os


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, batch_size=4, input_size=(800, 800, 3), output_size=(200, 200, 3), dataset_dir='src/dataset', shuffle=True, train = True):
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.dataset_dir = dataset_dir
        self.image_dir = os.path.join(dataset_dir, "images/")
        self.demo_dir = os.path.join(dataset_dir, "collage/")
        self.list_image = [x[0:-1] for x in open(os.path.join(dataset_dir, "image.txt")).readlines()]
        self.list_train = self.list_image[0:int(len(self.list_image)*0.9)]
        self.list_val = self.list_image[int(len(self.list_image)*0.9):]
        self.label = pd.read_csv(os.path.join(dataset_dir, "data.tsv"), sep="\t")
        self.shuffle = shuffle
        if train:
            self.list_image = self.list_train
        else:
            self.list_image = self.list_val
        self.on_epoch_end()

    # Denotes the number of batches per epoch
    def __len__(self):
        return int(np.floor(len(self.list_image) / self.batch_size))

    # Updates indexes after each epoch
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_image))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    # Generate one batch of data
    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of image and gt files
        list_image_tmp = [self.list_image[k] for k in indexes]             # list of image with their real photo_names
        list_gt_tmp = [eval(self.label[self.label.photo_name == (k+str(".jpg"))]["location"].values[0]) for k in list_image_tmp] # An list containing batch_size dictionaries
        
        # Generate data
        X, y = self.__data_generation(list_image_tmp, list_gt_tmp)

        return X, y

    # Generates data containing batch_size samples
    def __data_generation(self, list_image_tmp, list_gt_tmp):
        # Initialization
        X = np.empty((self.batch_size, *self.input_size))
        y = np.empty((self.batch_size, *self.output_size), dtype=float)

        # Generate data
        for i, image in enumerate(list_image_tmp):
            # Store sample
            image_path = os.path.join(self.image_dir + image + str(".jpg"))
            img = cv2.imread(image_path)

            img = cv2.resize(img, (self.input_size[0], self.input_size[1]))
            # plt.imshow(img[:,:,::-1])
            # plt.show()
            X[i,] = img
            # Store heatmaps
            gt = list_gt_tmp[i]["data"]
            heatmap = generate_center_heatmap(self.output_size, gt)
            y[i,] = heatmap
        return X, y
    

    def getitem(self, index):
        image = self.list_image[index]
        image_path = os.path.join(self.demo_dir + image + str(".jpg"))
        img = cv2.imread(image_path)
        return img



        

if __name__ == '__main__':
    dataloader = DataGenerator()
    dataloader.__getitem__(0)
    # dataloader.getitem(0)
    
