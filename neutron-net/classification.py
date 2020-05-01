from comet_ml import Experiment

import os, time, re, glob, warnings
import argparse
import json

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

def main(args):
    pass

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