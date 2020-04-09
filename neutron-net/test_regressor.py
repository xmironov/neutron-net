from comet_ml import Experiment

import os 
os.environ["KMP_AFFINITY"] = "none"

import argparse
import h5py 
import time 
import re 
import json
import glob
import pickle
import warnings
import dataseq


import numpy as np 
import pandas as pd
import imgaug as ia 
import seaborn as sn
import matplotlib.pyplot as plt


from datetime import datetime
from sklearn.metrics import mean_squared_error, confusion_matrix
from imgaug import augmenters as iaa

from tensorflow.keras.models import Sequential, model_from_yaml
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam

import confusion_matrix_pretty_print

DIMS          = (300, 300)
CHANNELS      = 1

class DataSequenceValues(Sequence):
    ''' Use Keras sequence to load image data from h5 file '''
    def __init__(self, h5_file, values, dim, channels, batch_size, layers):
        'Initialisation'
        self.file       = h5_file                 # H5 file to read
        self.values     = values                  # Values of images  
        self.dim        = dim                     # Image dimensions
        self.channels   = channels                # Image channels                   
        self.batch_size = batch_size              # Batch size
        self.layers     = layers
        self.on_epoch_end()

    def __len__(self):
        'Denotes number of batches per epoch'
        return int(np.floor(len(self.file['images']) / self.batch_size))

    def __getitem__(self, index):
        'Generates one batch of data'
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        x, y = self.__data_generation(indexes)

        return x, y

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'
        x = np.empty((self.batch_size, *self.dim, self.channels))
        y = np.empty((self.batch_size, self.layers * 2), dtype=float)

        for i, idx in enumerate(indexes):
            x[i,] = np.array(self.file['images'][idx])
            y[i,] = self.values[idx]
        
        return x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.file['images']))

    def close_file(self):
        self.file.close()

def read_config(f):
    with open(f, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)

def read_history(f):
    with open(f, 'r') as stream:
        return json.load(stream)

def main(args):
    selected_model = select_directory(os.path.join(args.model, '*/'), 'model')
    model = get_model(selected_model[0])
    model.load_weights(os.path.join(selected_model[0], 'model_weights.h5'))
    selected_data = select_directory(os.path.join(args.data, '*/'), 'data')
    layers = int(os.path.basename(os.path.dirname(selected_data[0])))
    scaler = pickle.load(open(os.path.join(selected_model[0], 'output_scaler.p'), 'rb'))
    
    tot_outputs = None
    tot_predictions = None

    for dataset in selected_data:
        outputs_dict = load_output(dataset)
        remainder = len(outputs_dict['test_output']) % args.batch_size
        if remainder == 0:
            test_outputs = outputs_dict['test_output']
        else:
            test_outputs = outputs_dict['test_output'][:-remainder]
        test_outputs = trim_array(test_outputs, layers)



        h5_file = h5py.File(os.path.join(dataset, 'test.h5'), 'r')
        test_sequence = DataSequenceValues(
            h5_file, test_outputs, DIMS, CHANNELS, args.batch_size, layers
        )

        predictions_raw = model.predict_generator(generator=test_sequence, use_multiprocessing=False, verbose=1)
        # predictions_exp = np.c_[predictions_raw, np.zeros(len(predictions_raw)), np.zeros(len(predictions_raw))]
        predictions = scaler.inverse_transform(predictions_raw)

    fig, ax = plt.subplots(2,2, figsize=(15,10))
    ax[0,0].scatter(test_outputs[:,0], predictions[:,0], alpha=0.2)
    ax[0,0].set_title('Layer 1')
    ax[0,0].set_xlabel('Actual depth')
    ax[0,0].set_ylabel('Predicted depth')

    ax[0,1].scatter(test_outputs[:,2], predictions[:,2], alpha=0.2)
    ax[0,1].set_title('Layer 2')
    ax[0,1].set_xlabel('Actual depth')
    ax[0,1].set_ylabel('Predicted depth')

    ax[1,0].scatter(test_outputs[:,1], predictions[:,1], alpha=0.2)
    ax[1,0].set_xlabel('Actual SLD')
    ax[1,0].set_ylabel('Predicted SLD')

    ax[1,1].scatter(test_outputs[:,3], predictions[:,3], alpha=0.2)
    ax[1,1].set_xlabel('Actual SLD')
    ax[1,1].set_ylabel('Predicted SLD')

    ax[0,0].set_ylim(-100,3010)
    ax[0,0].set_xlim(-100,3010)
    ax[0,1].set_ylim(-100,3010)
    ax[0,1].set_xlim(-100,3010)

    ax[1,0].set_ylim(-0.1,1.1)
    ax[1,0].set_xlim(-0.1,1.1)
    ax[1,1].set_ylim(-0.1,1.1)
    ax[1,1].set_xlim(-0.1,1.1)

    plt.tight_layout()
    plt.show()

    

        
    #     test_sequence = DataSequenceClasses(
    #         os.path.join(dataset, 'perfect_test.h5'), test_classes, DIMS, CHANNELS, BATCH_SIZE)
    #     layer_predictions_raw = model.predict_generator(generator=test_sequence, use_multiprocessing=False, verbose=1, workers=0)
    #     layer_predictions = np.argmax(layer_predictions_raw, axis=1)

    #     if (tot_classes is None) & (tot_predictions is None):
    #         tot_classes = test_classes
    #         tot_predictions = layer_predictions
    #     else:
    #         tot_classes = np.concatenate((tot_classes, test_classes), axis=0)
    #         tot_predictions = np.concatenate((tot_predictions, layer_predictions), axis=0)

    # cm = confusion_matrix(tot_classes, tot_predictions)
    # df_cm = pd.DataFrame(cm, index=[i for i in '12'], columns=[i for i in '12'])
    # confusion_matrix_pretty_print.pretty_plot_confusion_matrix(df_cm)


def load_output(path):
    data = {}
    for section in ['test']:
        with h5py.File(os.path.join(path, 'perfect_{}.h5'.format(section)), 'r') as f:
            data['{}_output'.format(section)] = np.array(f['Y'])
    return data

def select_directory(path, itype):
    dirs = glob.glob(path)
    print('''\n ### Select %s: ###''' % itype)
    for i, directory in enumerate(dirs):
        print(str(i + 1) + " " + directory)
    if itype == 'data':
        print('''\n If you would like to predict on all the layers, please type: all''')
    directory_idx = input("-> ")
    
    if directory_idx == 'all':
        print('\n Selected: [all layers]')
        return dirs

    directory_path = dirs[int(directory_idx) - 1]
    print("\n Selected: " + directory_path + "\n")

    return [directory_path]

def get_model(path):
    yaml_file = open(os.path.join(path, 'model.yaml'), 'r')
    model_yaml = yaml_file.read()
    yaml_file.close()
    return model_from_yaml(model_yaml)

def trim_array(array, layers):
    ''' For expected number of layers, trims array to give expected size '''
    actual_size = array.shape[1]
    expected_size = layers * 2
    difference = actual_size - expected_size

    if difference == 0:
        return array
    
    return np.delete(array, np.s_[-difference:], axis=1)

def parse():
    parser = argparse.ArgumentParser(description='PyTorch RefNet Training')
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('model', metavar='MODEL',
                        help='path to directory of models')
    parser.add_argument('--all', dest='all', action='store_true', help='Predict on all layers')
    parser.add_argument('-b', '--batch_size', default=64, type=int,
                        metavar='N', help='mini-batch size per process (default: 64)')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse()
    main(args)