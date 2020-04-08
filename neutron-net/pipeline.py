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

from sequencers import DataSequence

DIMS          = (300, 300)
CHANNELS      = 1

def create_save_directories(data):
    savepaths = {}
    name = 'refnet_pipe-' + datetime.now().strftime('%Y-%m-%dT%H%M%S')
    directories = ['img', 'fits', 'predictions', 'results']
    for directory in directories:
        if directory == 'results':
            directory_path = os.path.join(data, directory)
        else:
            directory_path = os.path.join(data, directory, name)

        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        savepaths[directory] = directory_path
    return savepaths
        
def main(args):
    # Directory setup
    data = os.path.normpath(args.data)
    savepaths = create_save_directories(data)

    # Create necessary numpy image files
    files = glob.glob(os.path.join(data, '*.dat'))
    n_files = len(files)
    image_filenames = create_images_from_directory(files, args.data, npy_savedir)
    class_labels = dict(zip(image_filenames, np.zeros((len(image_filenames), 1))))

    # Generator to yield numpy image files
    classification_test_loader = DataSequence(
        DIMS, CHANNELS, batch_size=1, mode='classification', labels=class_labels)
    )
    
    # Loading classifier model
    classifier_model = get_model(args.classifier_model)
    classifier_model.load_weights(os.path.join(args.classifier_model, 'model_weights.h5'))
    test_classification_predictions = classifier_model.predict(classification_test_loader, verbose=1)
    test_classification_predictions = np.argmax(test_classification_predictions, axis=1)

    # classification_predictions = classifier_model.predict(test_loader, verbose=1)
    fake_classification_predictions = np.full((n_files, 1), 2)

    values_labels = {filename: {'depth': np.zeros((1,int(prediction))), 'sld': np.zeros((1,int(prediction))), 'class': int(prediction)}
                        for filename, prediction in zip(image_filenames, fake_classification_predictions)}

    regression_test_loader = DataSequence(
        DIMS, CHANNELS, batch_size=1, mode='regression', labels=values_labels)
    )

    # TODO: sort out the regression end
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
                        help='path to directory of .dat files')
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