import os 
os.environ["KMP_AFFINITY"] = "none"

import argparse
import h5py, time, re, sys 
import json, glob, pickle, random
import warnings
import pathlib
import shutil

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from datetime import datetime
from sklearn.metrics import mean_squared_error
from skimage import data, color

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_yaml

DIMS = (300, 300)
CHANNELS = 1
tf.compat.v1.disable_eager_execution()

class DataLoaderClassification(Sequence):
    ''' Use Keras sequence to load image data from h5 file '''
    def __init__(self, labels, dim, channels, batch_size):
        'Initialisation'
        self.labels      = labels                

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
        indexes = [list(self.labels.keys())[k] for k in indexes]
        images, targets = self.__data_generation(indexes)

        return images, targets

    def __data_generation(self, indexes):
        # 'Generates data containing batch_size samples'
        images = np.empty((self.batch_size, *self.dim, self.channels))
        classes = np.empty((self.batch_size, 1), dtype=int)

        for i, np_image_filename in enumerate(indexes):
            images[i,] = np.load(np_image_filename)
            classes[i,] = self.labels[np_image_filename]
        
        return images, classes

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.labels.keys()))

    def close_file(self):
        self.file.close()

class DataLoaderRegression(Sequence):
    ''' Use Keras sequence to load image data from a dictionary '''
    def __init__(self, labels_dict, dim, channels):
        'Initialisation'
        self.labels_dict = labels_dict            
        self.dim        = dim                     # Image dimensions
        self.channels   = channels                # Image channels                   
        self.on_epoch_end()

    def __len__(self):
        'Denotes number of batches per epoch'
        # batch_size set as 1
        return int(np.floor(len(self.labels_dict.keys()) / 1))

    def __getitem__(self, index):
        'Generates one batch of data'
        indexes = self.indexes[index: (index + 1)]
        indexes = [list(self.labels_dict.keys())[k] for k in indexes]
        images = self.__data_generation(indexes)
        return images

    def __data_generation(self, indexes):
        # 'Generates data containing batch_size samples'
        images = np.empty((1, *self.dim, self.channels))
        layers = np.empty((1, 1))

        for i, np_image_filename in enumerate(indexes):
            images[i,] = np.load(np_image_filename)
            layers[i,] = self.labels_dict[np_image_filename]["class"]

        return images, layers

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.labels_dict.keys()))
    
class KerasDropoutPredicter():
    """ Class that takes trained models and uses Dropout at test time to make Bayesian-like predictions"""
    def __init__(self, models):
        # One-layer model function
        self.f_1 = K.function([models[1].layers[0].input, K.learning_phase()],
                              [models[1].layers[-2].output, models[1].layers[-1].output])

        # Two-layer model function
        self.f_2 = K.function([models[2].layers[0].input, K.learning_phase()],
                              [models[2].layers[-2].output, models[2].layers[-1].output])

    def predict(self, sequence, n_iter=5):
        steps_done = 0
        all_out = []
        steps = len(sequence)
        output_generator = iter_sequence_infinite(sequence)

        while steps_done < steps:
            # Yield the sample image, and the number of layers it is predicted to have
            x, y = next(output_generator)
            results = []

            for i in range(n_iter):
                # If one-layer
                if y[0][0] == 1:
                    result = self.f_1([x, 1])
                # else if two-layer
                elif y[0][0] == 2:
                    result = self.f_2([x, 1])
                results.append(result)

            results = np.array(results)
            prediction, uncertainty = results.mean(axis=0), results.std(axis=0)
            outs = np.array([prediction, uncertainty])

            if not all_out:
                for out in outs:
                    all_out.append([])

            for i, out in enumerate(outs):
                all_out[i].append(out)
            
            steps_done+=1
        return [np.concatenate(out, axis=1) for out in all_out]

def main(args):
    scaler_path = os.path.join(args.data, "output_scaler.p")
    save_paths = create_save_directories(args.data)

    dat_files, npy_image_filenames = dat_files_to_npy_images(args.data, save_paths["img"])
    class_labels = dict(zip(npy_image_filenames, np.zeros((len(npy_image_filenames), 1))))
    classifier_loader = DataLoaderClassification(class_labels, DIMS, CHANNELS, 1)

    try:
        scaler = pickle.load(open(scaler_path, "rb"))
    except OSError:
        print("output_scaler.p file not found in data directory provided")
    
    # Load models directly from paths
    classifier = load_model(args.classifier)
    one_layer = load_model(args.one_layer)
    two_layer = load_model(args.two_layer)
    models = {1: one_layer, 2: two_layer}

    # No. layer predictions
    layer_predictions = np.argmax(classifier.predict(classifier_loader, verbose=1), axis=1)
    
    # Dictionary to pair image with "depth", "sld", "class" values
    values_labels = {filename: {"depth": np.zeros((1, int(layer_prediction))),
                                "sld": np.zeros((1, int(layer_prediction))),
                                "class": int(layer_prediction)}
                        for filename, layer_prediction in zip(npy_image_filenames, layer_predictions)}

    loader = DataLoaderRegression(values_labels, DIMS, CHANNELS)
    
    # Use custom class to activate Dropout at test time in models
    kdp = KerasDropoutPredicter(models)
    kdp_predictions = kdp.predict(loader, n_iter=1)

    # Predictions given as [depth_1, depth_2], [sld_1, sld_2]
    depth_predictions, sld_predictions = kdp_predictions[0][0], kdp_predictions[0][1]

    # Errors given as [depth_std_1, depth_std_2], [sld_std_1, sld_std_2]
    depth_error, sld_error = kdp_predictions[1][0], kdp_predictions[1][1]

    ##########################################################################################################
    ## Current scaler expects 4-dimensional data of shape [depth_1, sld_1, depth_2, sld_2]
    ## Order of entries in data in array must be arranged to order expected by scaler
    ## If one-layer data, array must be expanded to [depth_1, sld_1, 0, 0] to keep same dimension as scaler
    ## If two-layer data, array does not need expanding, but still needs rearranging
    ##########################################################################################################
    frmt_predictions = []
    frmt_errors = []
    for d_pred, s_pred, d_error, s_error in zip(depth_predictions, sld_predictions, depth_error, sld_error):
        # If data is for one-layer expand matrix with zeros
        if (len(d_pred) == 1) & (len(s_pred) == 1):
            frmt_pred = np.c_[d_pred[0], s_pred[0], np.zeros(len(d_pred)), np.zeros(len(s_pred))]
            frmt_error = np.c_[d_error[0], s_error[0], np.zeros(len(d_pred)), np.zeros(len(s_pred))]
            
        elif (len(d_pred) == 2) & (len(s_pred) == 2):
            frmt_pred = np.c_[d_pred[0], s_pred[0], d_pred[1], s_pred[1]]
            frmt_error = np.c_[d_error[0], s_error[0], d_error[1], s_error[1]]
        
        frmt_predictions.append(frmt_pred)
        frmt_errors.append(frmt_error)
    
    frmt_predictions = np.vstack(frmt_predictions)
    frmt_errors = np.vstack(frmt_errors)
    
    scaled_predictions = scaler.inverse_transform(frmt_predictions)
    scaled_errors = scaler.inverse_transform(frmt_errors)

    for file, prediction in zip(dat_files, scaled_predictions):
        prediction = prediction.tolist()
        prediction[1], prediction[2] = prediction[2], prediction[1]
        prediction.insert(2, 0)
        dat_to_genX(save_paths["fits"], file, prediction)


def create_save_directories(data):
    """Takes a data path and creates a save directory structure for .npy images, fits and etc."""
    savepaths = {}
    name = 'Dev-' + datetime.now().strftime('%Y-%m-%dT%H%M%S')
    directories = ['img', 'fits', 'predictions', 'results']

    for directory in directories:
        # General results directory for all runs
        if directory == 'results':
            directory_path = os.path.join(data, directory)

        # Specific run directories
        else:
            directory_path = os.path.join(data, directory, name)

        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        savepaths[directory] = directory_path
    return savepaths

def dat_files_to_npy_images(data_path, save_path):
    """Locate any .dat files in given data_path, create .npy images and save them in save_path"""
    dat_files = glob.glob(os.path.join(data_path, '*.dat'))
    image_filenames = create_images_from_directory(dat_files, save_path)
    return dat_files, image_filenames

def create_images_from_directory(dat_files, save_path):
    """Take list of .dat files, creat .npy images and save them in savepath"""
    image_files = []

    for dat_file in dat_files:
        # Identify if there are column headings, or whether the header is empty
        header_setting = identify_header(dat_file)

        if header_setting is None:
            data = pd.read_csv(dat_file, header=0, delim_whitespace=True, names=['X', 'Y', 'Error'])
        else:
            data = pd.read_csv(dat_file, header=header_setting)

        head, tail = os.path.split(dat_file)
        name = os.path.normpath(os.path.join(save_path, tail)).replace(".dat", ".npy")
        image_files.append(name)
        sample_momentum = data["X"]
        sample_reflect = data["Y"]
        sample = np.vstack((sample_momentum, sample_reflect)).T
        img = image_process(sample)
        np.save(name, img)

    return image_files

def identify_header(path, n=5, th=0.9):
    """Parse the .dat file header to find out if there are headings or if it's empty"""
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
    plt.ylim(1e-08,1.5) #this hadnt been set previously!
    plt.axis("off")
    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    mplimage = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    gray_image = color.rgb2gray(mplimage)
    plt.close()
    return gray_image

def iter_sequence_infinite(sequence):
    """Infinite iterator for sequence"""
    while True:
        for item in sequence:
            yield item

def dat_to_genX(directory, filename, parameters=[50,50,0,0.1,0.1]):
    """ Copy base GenX file and populate the data and parameters in it """
    path = os.path.split(filename)
    name = os.path.splitext(path[1])
    hdf_name = str(name[0]) + "_starting_fit.hgx"

    base_directory = os.getcwd()
    copy_in_place(directory, base_directory, "GenX_Base_config.hgx", hdf_name)

    [x, y, e] = extract_data(filename)
    headings = ["x_raw", "y_raw", "error_raw", "x", "y", "error"]
    overwriting_data = [x, y, e, x, y, e]

    hdf_path =  os.path.join(directory, hdf_name)
    hdf_path = os.path.normpath(hdf_path)
    # hdf_read(hdf_path, 'current/parameters/data col 1')

    for heading, datum in zip(headings, overwriting_data):
        hdf_overwrite(hdf_path, 'current/data/datasets/0//' + str(heading), datum)

    for i in range(len(parameters)):
        hdf_overwrite(hdf_path, 'current/parameters/data col 1', parameters[i], single_value=True, index=i)
        hdf_overwrite(hdf_path, 'current/parameters/data col 3', parameters[i]*0.1, single_value=True, index=i)
        hdf_overwrite(hdf_path, 'current/parameters/data col 4', parameters[i]*3, single_value=True, index=i)

    # hdf_read(hdf_path, 'current/parameters/data col 1')

def hdf_read(file_name, group):
    try:
        f1 = h5py.File(file_name,'r')
        print(f1[str(group)][()])
        return f1[str(group)][()]
    finally:
        f1.close()       

def hdf_overwrite(file_name, group, new_data, single_value=False, index=0):
    try:
        if not single_value: #This will delete the group and then remake it, since there is no guarentee that it will be the same size
            f1 = h5py.File(str(file_name),'r+')
            del f1[str(group)]
            f1.close()

            f1 = h5py.File(file_name, 'r+')  
            foo = f1.create_dataset(str(group),data=new_data)       
        else:
            f1 = h5py.File(file_name, 'r+')     #This will just write in place as it is acessing a single value of array
            f1[str(group)][index] = new_data
    finally:
        f1.close() 

def copy_in_place(directory, base_directory, original_file, new_file_name):
    """This copies a file into a temporary location, and then renames it and moves it back, keeping the original intact
    It should be greral, but some file meta-data may be lost by shutil.copy
    It will overwrite any files with the same name as new_file_name in the original folder"""
    try:
        extension = os.path.splitext(original_file)[1] # Get the file extension in order to preserve it
        temp_folder = os.path.normpath(os.path.join(directory, "temp"))
        temp_file_name = os.path.join(temp_folder, "temp"+extension)
        full_filepath_orig = os.path.join(base_directory, original_file)
        full_filepath_temp = os.path.join(temp_folder, new_file_name)
        full_filepath_new = os.path.normpath(os.path.join(directory, new_file_name))
        move(full_filepath_orig, temp_file_name) # Copy the file into a temporary place to avoid overwriting the existing file

        os.rename(temp_file_name, full_filepath_temp) # Rename the file to the new name

        move(full_filepath_temp, full_filepath_new) # Move the newly renamaed file back into the folder
    
    finally:
        try:
            os.remove(full_filepath_temp) # Remove the temporary file and folder
            os.rmdir(temp_folder)
        except:
            pass

def move(old_file_name, new_file_name):
    dst_dir = os.path.dirname(new_file_name)
    if not os.path.exists(dst_dir): #If it isn't there, make it there
        pathlib.Path(str(dst_dir)).mkdir(exist_ok=True) 
    try: 
        shutil.copy(old_file_name,new_file_name)
    except (PermissionError, FileNotFoundError): # There is a race condition in here somewhere and I can't track it down, so this "fixes" it
        if not os.path.exists(dst_dir): #If it isn't there, make it there
            pathlib.Path(str(dst_dir)).mkdir(exist_ok=True)
            shutil.copy(old_file_name,new_file_name)

def extract_data(filename):
    header_setting = identify_header(filename)
    if header_setting is None:
        data = pd.read_csv(filename, header=0, delim_whitespace=True, names=['X', 'Y', 'Error'])
    else:
        data = pd.read_csv(filename, header=header_setting)

    x_data = data["X"]
    y_data = data["Y"]
    e_data = data["Error"]
    return [x_data,y_data,e_data]

def parse():
    parser = argparse.ArgumentParser(description="Prediction Pipeline")
    parser.add_argument("data", metavar="datapath", help="path to data directory with .dat files")
    parser.add_argument("classifier", metavar="classifier", help="path to classifier full_model.h5")
    parser.add_argument("one_layer", metavar="one_layer", help="path to one-layer regression full_model.h5")
    parser.add_argument("two_layer", metavar="two_layer", help="path to two-layer regression full_model.h5")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse()
    main(args)