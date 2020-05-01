from comet_ml import Experiment

import os, time, re, glob, warnings
import argparse
import json
import h5py

os.environ["KMP_AFFINITY"] = "none"

import numpy as np 
import pandas as pd 

from datetime import datetime
from sklearn.metrics import mean_squared_error, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, LeakyReLU, Dropout, Input
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Nadam

DIMS = (300, 300)
CHANNELS = 1
tf.compat.v1.disable_eager_execution()

class Sequencer(Sequence):
    """ Use Keras Sequence class to load image data from h5 file"""
    def __init__(self, file, labels, dim, channels, batch_size, debug=False, shuffle=False):
        self.file       = file
        self.labels     = labels
        self.dim        = dim
        self.channels   = channels
        self.batch_size = batch_size
        self.debug      = debug
        self.on_epoch_end()

    def __len__(self)
        """ Denotes number of batches per epoch"""
        return int(np.floor(len(self.labels) / self.batch_size))

    def __getitem__(self, index):
        """ Generates one batch of data"""
        indexes = self.indexes[index * self.batch_size: (index+1) * self.batch_size]
        inputs, targets = self.__data_generation(indexes)
        return inputs, targets

    def __data_generation(self, indexes):
        """ Generates data containing batch_size samples"""
        images = np.empty((self.batch_size, *self.dim, self.channels))
        classes = np.empty((self.batch_size, 1), dtype=int)

        for i, idx in enumerate(indexes):
            image = self.file['images'][idx]
            images[i,] = image
            classes[i,] = self.labels[idx]

            if self.debug:
                i = 0
                while i < 10:
                    print(classes[i])
                    i+=1
                self.debug = False
            
            return images, classes
    
    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        indexes = np.arange(len(self.labels))

        if self.shuffle:
            self.indexes = np.random.shuffle(indexes)
        else:
            self.indexes = indexes

def main(args):
    name = "classifier-[" + datetime.now().strftime("%Y-%m-%dT%H%M%S") + "]"
    savepath = os.path.join(args.save, name)

    if args.log:
        experiment = Experiment(api_key="Qeixq3cxlTfTRSfJ2hyPlMWjk", project_name="general", workspace="xandrovich")

    train_dir = os.path.join(args.data, "train.h5")
    validate_dir = os.path.join(args.data, "validate.h5")
    test_dir = os.path.join(args.data, "test.h5")

    train_file = h5py.File(train_dir, "r")
    validate_dir = h5py.File(validate_dir, "r")
    test_dir = h5py.File(test_dir, "r")

    train_loader = DataSequence(DIMS, CHANNELS, args.batch_size)

def parse():
    parser = argparse.ArgumentParser(description="Keras Classifier Training")
    # Meta Parameters
    parser.add_argument("data", metavar="PATH", help="path to data directory")
    parser.add_argument("save", metavar="PATH", help="path to save directory")
    parser.add_argument("-l", "--log", action="store_true", help="boolean: log metrics to CometML?")

    # Model parameters
    parser.add_argument("-e", "--epochs", default=2, type=int, metavar="N", help="number of epochs")
    parser.add_argument("-b", "--batch_size", default=20, type=int, metavar="N", help="no. samples per batch (def:20)")
    parser.add_argument("-j", "--workers", default=6, type=int, metavar="N", help="no. data loading workers (def:6)")
    
    # Learning parameters
    parser.add_argument("-lr", "--learning_rate", default=0.0003, type=float, metavar="R", help="Nadam learning rate")
    parser.add_argument("-dr", "--dropout_rate", default=0.1, type=float, metavar="R", help="dropout rate" )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse()
    main(args)