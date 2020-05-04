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
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, LeakyReLU, Dropout, Input
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Nadam

import confusion_matrix_pretty_print

DIMS = (300, 300)
CHANNELS = 1
tf.compat.v1.disable_eager_execution()

class Sequencer(Sequence):
    """ Use Keras Sequence class to load image data from h5 file"""
    def __init__(self, file, labels, dim, channels, batch_size, layers, debug=False, shuffle=False):
        self.indexes    = np.where(np.array(file["class"]) == layers)[0]
        self.file       = file
        self.labels     = labels[self.indexes]
        self.dim        = dim
        self.channels   = channels
        self.batch_size = batch_size
        self.debug      = debug
        self.shuffle    = shuffle
        self.layers     = layers
        self.on_epoch_end()
    
    def __len__(self):
        """ Denotes number of batches per epoch"""
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, idx):
        """ Generates one batch of data"""
        indexes = self.indexes[idx*self.batch_size:(idx+1) * self.batch_size]
        inputs, targets = self.__data_generation(indexes)

        return inputs, targets

    def __data_generation(self, indexes):
        """ Generates data containing batch_size samples"""
        images = np.empty((self.batch_size, *self.dim, self.channels))
        classes = np.empty((self.batch_size, 1), dtype=int)

        for i, idx in enumerate(indexes):
            image = self.file["images"][idx]
            images[i,] = np.array(image)
            classes[i,] = self.labels[idx]      
        return images, classes
    
    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        indexes = np.where(np.array(self.file["class"]) == self.layers)[0]

        if self.shuffle:
            np.random.shuffle(indexes)

        self.indexes = indexes

def main(args):
    name = "regressor-%s[" % str(args.layers) + datetime.now().strftime("%Y-%m-%dT%H%M%S") + "]"
    save = r"C:\Users\mtk57988\stfc\neutron-net\neutron-net\models\investigate"
    base = r"C:\Users\mtk57988\stfc\neutron-net\neutron-net\models\investigate\classifier-[2020-05-03T104626]\full_model.h5"
    data = r"C:\Users\mtk57988\stfc\ml-neutron\neutron_net\data\perfect_w_classes\all"

    savepath = os.path.join(save, name)

    if args.log:
        experiment = Experiment(api_key="Qeixq3cxlTfTRSfJ2hyPlMWjk", project_name="general", workspace="xandrovich")
    
    train_dir = os.path.join(data, "train.h5")
    validate_dir = os.path.join(data, "valid.h5")
    test_dir = os.path.join(data, "test.h5")

    train_file = h5py.File(train_dir, "r")
    validate_file = h5py.File(validate_dir, "r")
    test_file = h5py.File(test_dir, "r")

    targets = load_targets(data)
    train_targets, validate_targets, test_targets = targets["train"], targets["valid"], targets["test"]

    train_loader = Sequencer(train_file, train_targets, DIMS, CHANNELS, args.batch_size, args.layers)
    validate_loader = Sequencer(validate_file, validate_targets, DIMS, CHANNELS, args.batch_size, args.layers)
    test_loader = Sequencer(test_file, test_targets, DIMS, CHANNELS, args.batch_size, args.layers)

def parse():
    parser = argparse.ArgumentParser(description="Keras Regressor Training")

    # Meta Parameters
    parser.add_argument("data", metavar="PATH", help="path to data directory")
    parser.add_argument("save", metavar="PATH", help="path to save directory")
    parser.add_argument("base", metavar="PATH", help="path to base classifier model")
    parser.add_argument("layers", metavar="N", type=int, help="no. layers of system")
    parser.add_argument("-l", "--log", action="store_true", help="boolean: log metrics to CometML?")
    parser.add_argument("-s", "--summary", action="store_true", help="show model summary")

    # Model parameters
    parser.add_argument("-e", "--epochs", default=2, type=int, metavar="N", help="number of epochs")
    parser.add_argument("-b", "--batch_size", default=40, type=int, metavar="N", help="no. samples per batch (def:40)")
    parser.add_argument("-j", "--workers", default=1, type=int, metavar="N", help="no. data loading workers (def:1)")

    # Learning parameters
    parser.add_argument("-lr", "--learning_rate", default=0.0003, type=float, metavar="R", help="Nadam learning rate")
    parser.add_argument("-dr", "--dropout_rate", default=0.1, type=float, metavar="R", help="dropout rate" )
    return parser.parse_args()

def load_targets(path):
    data = {}
    for section in ["train", "valid", "test"]:
        with h5py.File(os.path.join(path, "{}.h5".format(section)), "r") as f:
            data["{}".format(section)] = np.array(f["scaledY"])

    return data

if __name__ == "__main__":
    args = parse()
    main(args)