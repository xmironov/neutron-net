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


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


from datetime import datetime
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam, Nadam

class RefModelClassifier():
    def __init__(self, dims, channels, epochs, dropout, learning_rate, workers):
        'Initialisation'
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
            patience=5,
            verbose=1,
            factor=0.5,
            min_lr = 0.00001
        )

        start = time.time()
        self.history = self.model.fit_generator(
            generator = train_seq,
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
        model = Sequential()

        # Convolutional network
        model.add(Conv2D(32, kernel_size=(3,3), activation="relu", input_shape=(*self.dims, self.channels)))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(64, kernel_size=3, activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        model.add(Conv2D(32, kernel_size=3, activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(16, kernel_size=3, activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))

        # Dense network
        model.add(Flatten())
        model.add(Dense(units=300, activation="relu", kernel_initializer="he_normal"))
        model.add(Dropout(self.dropout))
        model.add(Dense(units=240, activation="relu", kernel_initializer="he_normal"))
        model.add(Dropout(self.dropout))
        model.add(Dense(units=192, activation="relu", kernel_initializer="he_normal"))
        model.add(Dropout(self.dropout))
        model.add(Dense(units=154, activation="relu", kernel_initializer="he_normal"))
        model.add(Dropout(self.dropout))
        model.add(Dense(units=123, activation="relu", kernel_initializer="he_normal"))
        model.add(Dropout(self.dropout))
        model.add(Dense(units=98, activation="relu", kernel_initializer="he_normal"))
        model.add(Dropout(self.dropout))
        model.add(Dense(units=79, activation="relu", kernel_initializer="he_normal"))
        model.add(Dropout(self.dropout))
        model.add(Dense(units=63, activation="relu", kernel_initializer="he_normal"))
        model.add(Dropout(self.dropout))
        model.add(Dense(units=50, activation="relu", kernel_initializer="he_normal"))
        model.add(Dropout(self.dropout))
        model.add(Dense(units=3, activation="softmax"))
        model.compile(
            optimizer = Nadam(self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics = ['sparse_categorical_accuracy']
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

class DataSequenceClasses(Sequence):
    ''' Use Keras sequence to load image data from h5 file '''
    def __init__(self, h5_file, classes, dim, channels, batch_size):
        'Initialisation'
        self.file       = h5py.File(h5_file, 'r') # H5 file to read
        self.classes    = classes                 # Classes of images  
        self.dim        = dim                     # Image dimensions
        self.channels   = channels                # Image channels                   
        self.batch_size = batch_size              # Batch size
        self.on_epoch_end()

    def __len__(self):
        'Denotes number of batches per epoch'
        return int(np.floor(len(self.file['images']) / self.batch_size))

    def __getitem__(self, index):
        'Generates one batch of data'
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        x, c = self.__data_generation(indexes)

        return x, c

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'
        x = np.empty((self.batch_size, *self.dim, self.channels))
        c = np.empty((self.batch_size, 1), dtype=int)

        for i, idx in enumerate(indexes):
            x[i,] = np.array(self.file['images'][idx])
            c[i,] = self.classes[idx]
        
        return x, c

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.file['images']))

    def close_file(self):
        self.file.close()
    
def main():
    experiment = Experiment(api_key="Qeixq3cxlTfTRSfJ2hyPlMWjk",
                            project_name="general", workspace="xandrovich")

    LEARNING_RATE = 0.0003
    DROPOUT       = 0.1
    BATCH_SIZE    = 40
    DIMS          = (300, 300)
    CHANNELS      = 1
    EPOCHS        = 100
    WORKERS       = 6

    datapath = r'C:\Users\mtk57988\stfc\ml-neutron\neutron_net\data\perfect_w_classes\all'
    savedir = r'C:\Users\mtk57988\stfc\ml-neutron\neutron_net\investigate\investigate'
    savepath = os.path.join(savedir, ('refnet-classifier-' + datetime.now().strftime('%Y-%m-%dT%H%M%S')))

    output_dict = load_output(datapath)
    classes_dict = load_classes(datapath)

    train_output, valid_output, test_output = output_dict['train_output'], output_dict['valid_output'], output_dict['test_output']
    train_classes, valid_classes, test_classes = classes_dict['train_class'], classes_dict['valid_class'], classes_dict['test_class']

    train_sequence = DataSequenceClasses(
        os.path.join(datapath, 'train.h5'), train_classes, DIMS, CHANNELS, BATCH_SIZE)
    valid_sequence = DataSequenceClasses(
        os.path.join(datapath, 'valid.h5'), valid_classes, DIMS, CHANNELS, BATCH_SIZE)
    test_sequence = DataSequenceClasses(
        os.path.join(datapath, 'test.h5'), test_classes, DIMS, CHANNELS, BATCH_SIZE)

    model = RefModelClassifier(DIMS, CHANNELS, EPOCHS, DROPOUT, LEARNING_RATE, WORKERS)
    model.summary()
    # model.plot()
    history = model.train(train_sequence, valid_sequence)
    model.save(savepath)
    
def load_output(path):
    data = {}
    for section in ['train', 'valid', 'test']:
        with h5py.File(os.path.join(path, '{}.h5'.format(section)), 'r') as f:
            data['{}_output'.format(section)] = np.array(f['Y'])
    return data

def load_classes(path):
    data = {}
    for section in ['train', 'valid', 'test']:
        with h5py.File(os.path.join(path, '{}.h5'.format(section)), 'r') as f:
            data['{}_class'.format(section)] = np.array(f['class'])
    return data

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
    main()