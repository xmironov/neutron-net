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
from sklearn.metrics import mean_squared_error, confusion_matrix
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

class RefModelClassifier():
    def __init__(self, dims, channels, epochs, dropout, learning_rate, workers, batch_size):
        'Initialisation'
        self.dims          = dims
        self.channels      = channels
        self.epochs        = epochs
        self.dropout       = dropout
        self.learning_rate = learning_rate
        self.workers       = workers
        self.batch_size    = batch_size
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

    def test(self, test_seq, file):
        layer_predictions = self.model.predict(test_seq, use_multiprocessing=False, verbose=1)
        layer_predictions = np.argmax(layer_predictions, axis=1)

        with h5py.File(file, 'r') as f:
            layers_ground = f['layers'] 
            remainder = len(layers_ground) % self.batch_size

            if remainder:
                layers_ground = layers_ground[:-remainder]

            cm = confusion_matrix(layers_ground, layer_predictions)
            df_cm = pd.DataFrame(cm, index=[i for i in '12'], columns=[i for i in '12'])
            # confusion_matrix_pretty_print.pretty_plot_confusion_matrix(df_cm)
    
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

        # Dense network
        dense_1 = Dense(300, activation='relu', kernel_initializer='he_normal')(flatten)
        dropout_1 = Dropout(self.dropout)(dense_1)
        dense_2 = Dense(240, activation='relu', kernel_initializer='he_normal')(dropout_1)
        dropout_2 = Dropout(self.dropout)(dense_2)
        dense_3 = Dense(192, activation='relu', kernel_initializer='he_normal')(dropout_2)
        dropout_3 = Dropout(self.dropout)(dense_3)
        dense_4 = Dense(154, activation='relu', kernel_initializer='he_normal')(dropout_3)
        dropout_4 = Dropout(self.dropout)(dense_4)
        dense_5 = Dense(123, activation='relu', kernel_initializer='he_normal')(dropout_4)
        dropout_5 = Dropout(self.dropout)(dense_5)
        dense_6 = Dense(98, activation='relu', kernel_initializer='he_normal')(dropout_5)
        dropout_6 = Dropout(self.dropout)(dense_6)
        dense_7 = Dense(79, activation='relu', kernel_initializer='he_normal')(dropout_6)
        dropout_7 = Dropout(self.dropout)(dense_7)
        dense_8 = Dense(63, activation='relu', kernel_initializer='he_normal')(dropout_7)
        dropout_8 = Dropout(self.dropout)(dense_8)
        dense_9 = Dense(50, activation='relu', kernel_initializer='he_normal')(dropout_8)
        dropout_9 = Dropout(self.dropout)(dense_9)
        dense_output = Dense(3, activation='softmax')(dropout_9)

        model = Model(inputs=input_img, outputs=dense_output)
        model.compile(optimizer=Nadam(self.learning_rate), 
                        loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

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
    savepath = os.path.join(
        args.save, 'classifier-[' + datetime.now().strftime('%Y-%m-%dT%H%M%S')  +']')

    if args.log:
        experiment = Experiment(api_key="Qeixq3cxlTfTRSfJ2hyPlMWjk",
                                project_name="general", workspace="xandrovich")

    traindir = os.path.join(args.data, 'train.h5')
    valdir = os.path.join(args.data, 'validate.h5')
    testdir = os.path.join(args.data, 'test.h5')

    trainh5 = h5py.File(traindir, 'r')
    valh5 = h5py.File(valdir, 'r')
    testh5 = h5py.File(testdir, 'r')

    train_loader = DataSequence(
        DIMS, CHANNELS, args.batch_size, mode='classification', h5_file=trainh5, debug=True)

    valid_loader = DataSequence(
        DIMS, CHANNELS, args.batch_size, mode='classification', h5_file=valh5)

    test_loader = DataSequence(
        DIMS, CHANNELS, args.batch_size, mode='classification', h5_file=testh5)

    model = RefModelClassifier(
        DIMS, CHANNELS, args.epochs, args.dropout, args.learning_rate, args.workers, args.batch_size)
    
    model.summary()
    history = model.train(train_loader, valid_loader)
    model.test(test_loader, testdir)
    model.save(savepath)

def parse():
    parser = argparse.ArgumentParser(description='Keras Classifier Training')
    parser.add_argument('data', metavar='DATADIR',
                        help='path to dataset')
    parser.add_argument('save', metavar='SAVEDIR',
                        help='path to save directory')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=40, type=int,
                        metavar='BATCH_SIZE', help='mini-batch size per process (default: 40)')
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