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
    ''' Use Keras sequence to load image data from h5 file '''
    def __init__(self, labels, dim, channels, batch_size, layers):
        'Initialisation'
        self.labels      = labels                

        self.dim        = dim                     # Image dimensions
        self.channels   = channels                # Image channels                   
        self.batch_size = batch_size              # Batch size
        self.layers     = layers

        self.on_epoch_end()

    def __len__(self):
        'Denotes number of batches per epoch'
        return int(np.floor(len(self.labels.keys()) / self.batch_size))

    def __getitem__(self, index):
        'Generates one batch of data'
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        indexes = [list(self.labels.keys())[k] for k in indexes]
        images = self.__data_generation(indexes)

        return images

    def __data_generation(self, indexes):
        # 'Generates data containing batch_size samples'
        images = np.empty((self.batch_size, *self.dim, self.channels))
        # targets_depth = []
        # targets_sld = []

        for i, np_image_filename in enumerate(indexes):
            images[i,] = np.load(np_image_filename)
            # targets_depth.append(self.labels[np_image_filename]["depth"])
            # targets_sld.append(self.labels[np_image_filename]["sld"])
            
        # , {'depth': np.array(targets_depth), 'sld': np.array(targets_sld)}
        return images

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.labels.keys()))

    def close_file(self):
        self.file.close()

class KerasDropoutPredicter():
    def __init__(self, model, sequence):
        self.f = K.function(
            [model.layers[0].input, K.learning_phase()], 
            [model.layers[-2].output, model.layers[-1].output])

    def predict(self, seq, n_iter=2):
        steps_done = 0
        all_out = []
        steps = len(seq)
        output_generator = iter_sequence_infinite(seq)

        while steps_done < steps:
            generator_output = next(output_generator)
            x = generator_output

            results = []
            for i in range(n_iter):
                result = self.f([x, 1])
                results.append(result)

            results = np.array(results)
            prediction, uncertainty = results.mean(axis=0), results.std(axis=0)
            outs = np.array([prediction, uncertainty])

            if not all_out:
                for out in outs:
                    all_out.append([])

            for i, out in enumerate(outs):
                all_out[i].append(out)

            steps_done += 1
        return [np.concatenate(out, axis=1) for out in all_out]

def main(args):
    # Necessary Paths
    scaler_path = os.path.join(args.data, "output_scaler.p")
    r_1_path = args.regression_1_layer
    r_2_path = args.regression_2_layer
    c_path = args.classification
    save_paths = create_save_directories(args.data)

    sys.exit()
    dat_files, npy_image_filenames = dat_files_to_npy_images(data_path, save_paths["img"])
    class_labels = dict(zip(npy_image_filenames, np.zeros((len(npy_image_filenames), 1))))

    classification_loader = DataLoaderClassification(class_labels, DIMS, CHANNELS, 1)
    classification_model = get_model(class_path)
    classification_model.load_weights(os.path.join(class_path, "model_weights.h5"))

    one_layer_model = load_model(one_layer_path)
    two_layer_model = load_model(two_layer_path)

    # layer_predictions = np.argmax(classification_model.predict(classification_loader, verbose=1), axis=1)

    # values_labels = {filename: 
    #                         {'depth': np.zeros((1,int(layer_prediction))), 
    #                          'sld': np.zeros((1,int(layer_prediction))), 
    #                          'class': int(layer_prediction)}                    
    #                     for filename, layer_prediction in zip(npy_image_filenames, test_classification_predictions)}

    regression_loader_one = DataLoaderRegression(class_labels, DIMS, CHANNELS, 1, 1)
    regression_loader_two = DataLoaderRegression(class_labels, DIMS, CHANNELS, 1, 2)
    # values_predictions = two_layer_model.predict(regression_loader_two, verbose=1)

    scaler = pickle.load(open(scaler_path, "rb"))

    # kdp_2_layer = KerasDropoutPredicter2(two_layer_model, regression_loader_two)
    kdp_1_layer = KerasDropoutPredicter(one_layer_model, regression_loader_one)

    # preds_2_layer = kdp_2_layer.predict(regression_loader_two, n_iter=100)
    # depth_2, sld_2 = preds_2_layer[0][0], preds_2_layer[0][1]
    # depth_std_2, sld_std_2 = preds_2_layer[1][0], preds_2_layer[1][1]
    # padded_preds_2 = np.c_[depth_2[:,0], sld_2[:,0], depth_2[:,1], sld_2[:,1]]
    # padded_error_2 = np.c_[depth_std_2[:,0], sld_std_2[:,0], depth_std_2[:,1], sld_std_2[:,1]]
    # error_2 = scaler.inverse_transform(padded_error_2)
    # preds_2 = scaler.inverse_transform(padded_preds_2)
    # preds_2 = preds_2

    preds_1_layer = kdp_1_layer.predict(regression_loader_one, n_iter=500)
    depth_1, sld_1 = preds_1_layer[0][0], preds_1_layer[0][1]
    depth_std_1, sld_std_1 = preds_1_layer[1][0], preds_1_layer[1][1]
    padded_preds_1 = np.c_[depth_1[:,0], sld_1[:,0], np.zeros(len(depth_1)), np.zeros(len(sld_1))]
    padded_error_1 = np.c_[depth_std_1[:,0], sld_std_1[:,0], np.zeros(len(depth_1)), np.zeros(len(sld_1))]
    error_1 = scaler.inverse_transform(padded_error_1)
    preds_1 = scaler.inverse_transform(padded_preds_1)
    preds_1[:,0] = preds_1[:,0] 
    preds_1[:,2] = 0
    preds_1[:,3] = 0

    for file, prediction in zip(dat_files, preds_1):
        prediction = prediction.tolist()
        prediction[1], prediction[2] = prediction[2], prediction[1]
        prediction.insert(2, 0)
        dat_to_genX(save_paths["fits"], file, prediction)

    # for file, prediction in zip(dat_files, preds_2):
    #     prediction = prediction.tolist()
    #     prediction[1], prediction[2] = prediction[2], prediction[1]
    #     prediction.insert(2, 0)
    #     dat_to_genX(save_paths["fits"], file, prediction)

    
    



def create_save_directories(data):
    'Create necessary directory structure for saving fits and figures'
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

        try:
            os.makedirs(directory_path)
        except OSError:
            print("Data path provided does not exist")

        savepaths[directory] = directory_path
    return savepaths

def dat_files_to_npy_images(data, savepath):
    dat_files = glob.glob(os.path.join(data, '*.dat'))
    image_filenames = create_images_from_directory(dat_files, data, savepath)
    return dat_files, image_filenames

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
    plt.ylim(1e-08,1.5) #this hadnt been set previously!
    plt.axis("off")
    # plt.show()
    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    #print(width,height)
    mplimage = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    gray_image = color.rgb2gray(mplimage)
    # plt.show()
    plt.close()
    return gray_image

def get_model(path):
    yaml_file = open(os.path.join(path, 'model.yaml'), 'r')
    model_yaml = yaml_file.read()
    yaml_file.close()
    return model_from_yaml(model_yaml)

def iter_sequence_infinite(seq):
    while True:
        for item in seq:
            yield item

def dat_to_genX(directory, filename, pars=[50,50,0,0.1,0.1]):
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

    for i in range(len(pars)):
        hdf_overwrite(hdf_path, 'current/parameters/data col 1', pars[i], single_value=True, index=i)
        hdf_overwrite(hdf_path, 'current/parameters/data col 3', pars[i]*0.1, single_value=True, index=i)
        hdf_overwrite(hdf_path, 'current/parameters/data col 4', pars[i]*3, single_value=True, index=i)

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
    '''This copies a file into a temporary location, and then renames it and moves it back, keeping the original intact
    It should be greral, but some file meta-data may be lost by shutil.copy
    It will overwrite any files with the same name as new_file_name in the original folder'''
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

    # Meta Parameters
    parser.add_argument("data", metavar="datapath", help="path to data directory with .dat files")
    parser.add_argument("-c", "--classification", metavar="PATH", help="path to classifier model")
    parser.add_argument("-r1", "--regression_1_layer", metavar="PATH", help="path to 1-layer regression model")
    parser.add_argument("-r2", "--regression_2_layer", metavar="PATH", help="path to 2-layer regression model")

    parser.add_argument("--test", metavar="PATH", help="path to regression model you wish to test")
    parser.add_argument("--bayesian", action="store_true", help="boolean: be bayesian?")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse()
    main(args)