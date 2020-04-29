from comet_ml import Experiment

import os 
os.environ["KMP_AFFINITY"] = "none"

import argparse
import h5py 
import time 
import re 
import json
import glob
import warnings
# import dataseq

import numpy as np 
import pandas as pd
# import imgaug as ia 

from datetime import datetime
from sklearn.metrics import mean_squared_error
# from imgaug import augmenters as iaa

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, LeakyReLU, BatchNormalization, Dropout, Input
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, Nadam

from sequencers import DataSequence

DIMS = (300, 300)
CHANNELS = 1
tf.compat.v1.disable_eager_execution()

class RefModelRegressor():
    def __init__(self, dims, channels, epochs, dropout, learning_rate, workers, layers):
        'Initialisation'
        self.outputs       = layers
        self.dims          = dims
        self.channels      = channels
        self.epochs        = epochs
        self.dropout       = dropout
        self.learning_rate = learning_rate
        self.workers       = workers
        self.model         = self.create_model()

    def train(self, train_seq, valid_seq):
        'Trains data on Sequences'

        # early_stop_cbk = EarlyStopping(
        #     monitor='val_loss',
        #     mode='min',
        #     patience=5,
        #     restore_best_weights=True,
        # )

        learning_rate_reduction_cbk = ReduceLROnPlateau(
            monitor='val_loss',
            patience=10,
            verbose=1,
            factor=0.5,
            min_lr = 0.000001
        )

        model_checkpoint_cbk = ModelCheckpoint(
            'weights.{epoch:02d}-{val_loss:2f}.h5',
            monitor='val_loss',
            verbose=0,
            save_best_only=True
        )

        start = time.time()
        self.history = self.model.fit(
            train_seq,
            validation_data = valid_seq,
            epochs = self.epochs,
            workers = self.workers,
            use_multiprocessing = False,
            verbose = 1,
            callbacks = [learning_rate_reduction_cbk]
        )

        elapsed_time = time.time() - start 
        self.time_taken = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

        train_seq.close_file()
        valid_seq.close_file()

        return self.history
    
    def create_model(self):
        # Convolutional Encoder
        input_img = Input(shape=(*self.dims, self.channels))
        conv_1 = Conv2D(32, (3,3), activation='relu')(input_img)
        pool_1 = MaxPooling2D((2,2))(conv_1)
        conv_2 = Conv2D(64, (3,3), activation='relu')(pool_1)
        pool_2 = MaxPooling2D((2,2), strides=(2,2))(conv_2)
        conv_3 = Conv2D(32, (3,3), activation='relu')(pool_2)
        pool_3 = MaxPooling2D((2,2))(conv_3)
        conv_4 = Conv2D(16, (3,3), activation='relu')(pool_3)
        pool_4 = MaxPooling2D((2,2))(conv_4)
        flatten = Flatten()(pool_4)

        # Depth feed-forward
        dense_1_d = Dense(units=300, activation='relu', kernel_initializer='he_normal')(flatten)
        dropout_1_d = Dropout(self.dropout)(dense_1_d)
        dense_2_d = Dense(units=192, activation='relu', kernel_initializer='he_normal')(dropout_1_d)
        dropout_2_d = Dropout(self.dropout)(dense_2_d)
        dense_3_d = Dense(units=123, activation='relu', kernel_initializer='he_normal')(dropout_2_d)
        dropout_3_d = Dropout(self.dropout)(dense_3_d)
        dense_4_d = Dense(units=79, activation='relu', kernel_initializer='he_normal')(dropout_3_d)
        dropout_4_d = Dropout(self.dropout)(dense_4_d)
        dense_5_d = Dense(units=50, activation='relu', kernel_initializer='he_normal')(dropout_4_d)
        dropout_5_d = Dropout(self.dropout)(dense_5_d)
        depth_linear = Dense(units=self.outputs, activation='linear', name='depth')(dropout_5_d)
        sld_linear = Dense(units=self.outputs, activation='linear', name='sld')(dropout_5_d)

        # SLD feed-forward
        # dense_1_SLD = Dense(units=300, activation='relu', kernel_initializer='he_normal')(flatten)
        # dropout_1_SLD = Dropout(self.dropout)(dense_1_SLD)
        # dense_2_SLD = Dense(units=192, activation='relu', kernel_initializer='he_normal')(dropout_1_SLD)
        # dropout_2_SLD = Dropout(self.dropout)(dense_2_SLD)
        # dense_3_SLD = Dense(units=123, activation='relu', kernel_initializer='he_normal')(dropout_2_SLD)
        # dropout_3_SLD = Dropout(self.dropout)(dense_3_SLD)
        # dense_4_SLD = Dense(units=79, activation='relu', kernel_initializer='he_normal')(dropout_3_SLD)
        # dropout_4_SLD = Dropout(self.dropout)(dense_4_SLD)
        # dense_5_SLD = Dense(units=50, activation='relu', kernel_initializer='he_normal')(dropout_4_SLD)
        # dropout_5_SLD = Dropout(self.dropout)(dense_5_SLD)
        # sld_linear = Dense(units=self.outputs, activation='linear')(dropout_5_SLD)

        model = Model(inputs=input_img, outputs=[depth_linear, sld_linear])
        model.compile(
            loss = {
                'depth':'mse',
                'sld'  :'mse',
            },
            loss_weights = {
                'depth': 1,
                'sld'  : 1,
            },
            optimizer = Nadam(self.learning_rate),
            metrics = {
                'depth':'mae',
                'sld'  :'mae',
            }
        )
        return model

    def summary(self):
        self.model.summary()

    def plot(self):
        plot_model(self.model, to_file='model.png')

    def save(self, savepath):
        try:
            os.makedirs(savepath)
            print('Created path: ' + savepath)
        except OSError:
            pass

        with open(os.path.join(savepath, 'history.json'), 'w') as f:
            json_dump = convert_to_float(self.history.history)
            json_dump['timetaken'] = self.time_taken
            json.dump(json_dump, f)

        model_yaml = self.model.to_yaml()

        with open(os.path.join(savepath, 'model.yaml'), 'w') as yaml_file:
            yaml_file.write(model_yaml)

        self.model.save_weights(os.path.join(savepath, 'model_weights.h5'))

        with open(os.path.join(savepath, 'summary.txt'), 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))

        self.model.save(os.path.join(savepath, 'full_model.h5'))
   
def main(args):
    if args.log:
        experiment = Experiment(api_key="Qeixq3cxlTfTRSfJ2hyPlMWjk",
                                project_name="general", workspace="xandrovich")

    layers = args.layers

    traindir = os.path.join(args.data, 'train.h5')
    valdir = os.path.join(args.data, 'validate.h5')
    testdir = os.path.join(args.data, 'test.h5')

    trainh5 = h5py.File(traindir, 'r')
    valh5 = h5py.File(valdir, 'r')
    testh5 = h5py.File(testdir, 'r')

    train_layers = np.array(trainh5['layers'])
    val_layers = np.array(valh5['layers'])
    test_layers = np.array(testh5['layers'])

    train_layers_indexes = np.where(train_layers==layers)[0]
    val_layers_indexes = np.where(val_layers==layers)[0]
    test_layers_indexes = np.where(test_layers==layers)[0]

    print(train_layers_indexes[0:5])

    # train_loader = DataSequence(
    #     DIMS, CHANNELS, args.batch_size, mode='regression', layers=args.layers, h5_file=trainh5)

    # valid_loader = DataSequence(
    #     DIMS, CHANNELS, args.batch_size, mode='regression', layers=args.layers, h5_file=valh5)

    # test_loader = DataSequence(
    #     DIMS, CHANNELS, args.batch_size, mode='regression', layers=args.layers, h5_file=testh5)

    # model = RefModelRegressor(
    #     DIMS, CHANNELS, args.epochs, args.dropout, args.learning_rate, args.workers, args.layers)
    # model.summary()
    # history = model.train(train_loader, valid_loader)
    # savepath = os.path.join(
    #     args.save, 'refnet-' + datetime.now().strftime('%Y-%m-%dT%H%M%S') + '-%slayer[keras]' % str(args.layers))
    # model.save(savepath)

def parse():
    parser = argparse.ArgumentParser(description='PyTorch RefNet Training')
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('save', metavar='SAVEDIR',
                        help='path to save directory')
    parser.add_argument('layers', metavar='LAYERS', type=int, choices=range(1,3),
                        help='number of layers to predict')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=64, type=int,
                        metavar='BATCH_SIZE', help='mini-batch size per process (default: 64)')
    parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                        help='number of data loading workers (default: 6)')
    parser.add_argument('-lr', '--learning_rate', default=0.0004, type=float,
                        help='initial learning rate')
    parser.add_argument('-d', '--dropout', default=0.1, type=float,
                        help='dropout rate')
    parser.add_argument('-l', '--log', action='store_true',
                        help='log metrics to CometML')   
    args = parser.parse_args()
    return args

def convert_to_float(dictionary):
	""" For saving model output to json"""
	jsoned_dict = {}
	for key in dictionary.keys():
		if type(dictionary[key]) == list:
			jsoned_dict[key] = [float(i) for i in dictionary[key]]
		else:
			jsoned_dict[key] = float(dictionary[key])
	return jsoned_dict

if __name__ == "__main__":
    args = parse()
    main(args)