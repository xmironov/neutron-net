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

import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime
from skimage import data, color
from tensorflow.keras.models import model_from_yaml
from tensorflow.keras.utils import Sequence

DIMS          = (300, 300)
CHANNELS      = 1

class DataSequenceClasses(Sequence):
    ''' Use Keras sequence to load image data from h5 file '''
    def __init__(self, labels, dim, channels, batch_size):
        'Initialisation'
        self.labels     = labels
        self.dim        = dim                     # Image dimensions
        self.channels   = channels                # Image channels                   
        self.batch_size = batch_size              # Batch size
        self.on_epoch_end()

    def __len__(self):
        'Denotes number of batches per epoch'
        return int(np.floor(len(self.labels.keys()) / self.batch_size))

    def __getitem__(self, index):
        'Generates one batch of data'
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        np_image_batch = [list(self.labels.keys())[k] for k in indexes]
        x, c = self.__data_generation(np_image_batch)

        return x, c

    def __data_generation(self, np_image_batch):
        'Generates data containing batch_size samples'
        x = np.empty((self.batch_size, *self.dim, self.channels))
        c = np.empty((self.batch_size, 1), dtype=int)

        for i, np_image in enumerate(np_image_batch):
            x[i,] = np.load(np_image)
            c[i,] = self.labels[np_image]
        
        return x, c        

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.labels.keys()))

    def close_file(self):
        self.file.close()

class DataSequenceValues(Sequence):
    ''' Use Keras sequence to load image data from h5 file '''
    def __init__(self, labels, dim, channels, batch_size):
        'Initialisation'
        self.labels     = labels
        self.dim        = dim                     # Image dimensions
        self.channels   = channels                # Image channels                   
        self.batch_size = batch_size              # Batch size
        self.on_epoch_end()

    def __len__(self):
        'Denotes number of batches per epoch'
        return int(np.floor(len(self.labels.keys()) / self.batch_size))

    def __getitem__(self, index):
        'Generates one batch of data'
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        np_image_batch = [list(self.labels.keys())[k] for k in indexes]
        x, y = self.__data_generation(np_image_batch)

        return x, y

    def __data_generation(self, np_image_batch):
        'Generates data containing batch_size samples'
        x = np.empty((self.batch_size, *self.dim, self.channels))
        y_depth = []
        y_sld = []

        for i, np_image in enumerate(np_image_batch):
            x[i,] = np.load(np_image)
            y_depth.append(self.labels[np_image]['depth'])
            y_sld.append(self.labels[np_image]['sld'])

        return x, {'depth': np.array(y_depth), 'sld': np.array(y_sld)}     

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.labels.keys()))

    def close_file(self):
        self.file.close()

def main(args):
    # Directory setup
    data = os.path.normpath(args.data)
    name = 'refnet_cr-' + datetime.now().strftime('%Y-%m-%dT%H%M%S') + '[keras]'
    npy_savedir = os.path.join(data, 'img', name)
    genx_savedir = os.path.join(data, 'fits', name)
    preds_savedir = os.path.join(data, 'predictions', name)
    results_savedir = os.path.join(data, 'results')

    for savedir in [npy_savedir, genx_savedir, preds_savedir, results_savedir]:
        if not os.path.exists(savedir):
            os.makedirs(savedir)

    # Create necessary numpy image files
    files = glob.glob(os.path.join(data, '*.dat'))
    n_files = len(files)
    image_filenames = create_images_from_directory(files, args.data, npy_savedir)
    class_labels = dict(zip(image_filenames, np.zeros((len(image_filenames), 1))))

    # Generator to yield numpy image files
    classification_test_loader = DataSequenceClasses(
        class_labels, DIMS, CHANNELS, batch_size=1)
    
    # Loading classifier model
    classifier_model = get_model(args.classifier_model)
    classifier_model.load_weights(os.path.join(args.classifier_model, 'model_weights.h5'))
    test_classification_predictions = classifier_model.predict(classification_test_loader, verbose=1)
    test_classification_predictions = np.argmax(test_classification_predictions, axis=1)

    # classification_predictions = classifier_model.predict(test_loader, verbose=1)
    fake_classification_predictions = np.full((n_files, 1), 2)

    values_labels = {filename: {'depth': np.zeros((1,int(prediction))), 'sld': np.zeros((1,int(prediction))), 'class': int(prediction)}
                        for filename, prediction in zip(image_filenames, fake_classification_predictions)}

    regression_test_loader = DataSequenceValues(
        values_labels, DIMS, CHANNELS, batch_size=1
    )

    # #TODO: Load all regression models and select appropriate one on a per case basis
    top_level_regression_dir = args.regressor_model
    # path_to_one_layer_regression_model = glob.glob(os.path.join(top_level_regression_dir, str(1), '*/'))[0]
    path_to_two_layer_regression_model = glob.glob(os.path.join(top_level_regression_dir, str(2), '*/'))[0]

    # one_layer_regression_model = get_model(path_to_one_layer_regression_model)
    two_layer_regression_model = get_model(path_to_two_layer_regression_model)

    models = {
        # 1: one_layer_regression_model,
        2: two_layer_regression_model,
    }

    predictions = []
    for img_filename, labels in values_labels.items():
        img = np.expand_dims(np.load(img_filename), axis=0)
        img_prediction = models[labels['class']].predict(img)
        predictions.append(img_prediction)

    predictions = {img_filename: prediction for img_filename, prediction in zip(image_filenames, predictions)}

    # for image_filename in image_filenames:
    #     img = np.expand_dims(np.load(image_filename), axis=0)
    #     img_prediction = two_layer_regression_model.predict(img)
    #     print(img_prediction)

    # # # Loading regression model
    # regressor_model = get_model(args.regressor_model)
    # regressor_model.load_weights(os.path.join(args.regressor_model, 'model_weights.h5'))
    # regression_predictions = regressor_model.predict(regression_test_loader, verbose=1)
    # # # regression_scaler = pickle.load(open(os.path.join(args.regressor_model, 'output_scaler.p'), 'rb'))

def create_images_from_directory(files, datapath, savepath):
    image_files = []

    for file in files:
        header_setting = identify_header(file)
        if header_setting is None:
            data = pd.read_csv(file, header=0, delim_whitespace=True, names=['X', 'Y', 'Error'])
        else:
            data = pd.read_csv(file, header=header_setting)

        head, tail = os.path.split(file)
        name = os.path.normpath(os.path.join(savepath, tail)).replace(".dat", ".npy")
        image_files.append(name)
        sample_momentum = data["X"]
        sample_reflect = data["Y"]
        sample = np.vstack((sample_momentum, sample_reflect)).T
        img = image_process(sample)
        np.save(name, img)

    return image_files

def identify_header(path, n=5, th=0.9):
    df1 = pd.read_csv(path, header='infer', nrows=n)
    df2 = pd.read_csv(path, header=None, nrows=n)
    sim = (df1.dtypes.values == df2.dtypes.values).mean()
    return 'infer' if sim < th else None

def image_process(sample):
    """Return resized np array of image size (300,300,1)"""
    x = sample[:,0]
    y = sample[:,1]
    image = get_image(x,y)
    return(np.resize(image, (300, 300, 1)))

def get_image(x,y):
    """Plot image using matplotlib and return as array"""
    fig = plt.figure(figsize=(3,3))
    plt.plot(x,y)
    plt.yscale("log", basey=10)
    plt.xlim(0,0.3)
    plt.ylim(10e-08,1.5) #this hadnt been set previously!
    plt.axis("off")
    #plt.show()
    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    #print(width,height)
    mplimage = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    gray_image = color.rgb2gray(mplimage)
    plt.close()
    return gray_image

def parse():
    parser = argparse.ArgumentParser(description='PyTorch RefNet Training')
    parser.add_argument('data', metavar='DATA',
                        help='path to data directory')
    # parser.add_argument('save', metavar='SAVE',
    #                     help='path to save directory')
    parser.add_argument('classifier_model', metavar='CLASSIFIER MODEL',
                        help='path to model for classification')
    parser.add_argument('regressor_model', metavar='REGRESSOR MODEL',
                        help='path to model for regression')
    args = parser.parse_args()
    return args

def get_model(path):
    yaml_file = open(os.path.join(path, 'model.yaml'), 'r')
    model_yaml = yaml_file.read()
    yaml_file.close()
    return model_from_yaml(model_yaml)

if __name__ == "__main__":
    args = parse()
    main(args)