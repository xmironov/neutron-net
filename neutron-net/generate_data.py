import os
import sys
import glob
import h5py
import pickle
import random

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from numpy.random import seed
from sklearn.preprocessing import MinMaxScaler
from skimage import data, color

def main():
    savedirs = [r'D:\Users\Public\Documents\stfc\ml-neutron\neutron_net\data\perfect_dynamic_scaler\1',
                 r'D:\Users\Public\Documents\stfc\ml-neutron\neutron_net\data\perfect_dynamic_scaler\2']

    dat_files_dir = r'C:\Users\mtk57988\stfc\ml-neutron\neutron_net\data\neutron_data\simulated_data'

    one_layer_files = glob.glob(os.path.join(dat_files_dir, 'O') + '*')
    two_layer_files = glob.glob(os.path.join(dat_files_dir, 'Tw') + '*')
    
    layers_dict = {
        1: load_simulated_files_with_classes(one_layer_files, 1),
        2: load_simulated_files_with_classes(two_layer_files, 2),
    }

    for layers, data in layers_dict.items():
        length = data['targets'].shape[1]
        difference = length - layers * 2

        if difference:
            data['targets'] = data['targets'][:,:-difference]
    
    split_ratios = {
        'train': 0.8,
        'validate': 0.1,
        'test': 0.1
    }

    one_layer_training_split = splitter(layers_dict[1], split_ratios)
    two_layer_training_split = splitter(layers_dict[2], split_ratios)

    shuffler(one_layer_training_split)
    shuffler(two_layer_training_split)

    one_layer_output_scaler = scale_targets(one_layer_training_split)
    two_layer_output_scaler = scale_targets(two_layer_training_split)
    pickle.dump(one_layer_output_scaler, open(os.path.join(savedirs[0], 'output_scaler.p'), 'wb'))
    pickle.dump(two_layer_output_scaler, open(os.path.join(savedirs[1], 'output_scaler.p'), 'wb'))
    scale_inputs(one_layer_training_split)
    scale_inputs(two_layer_training_split)

    assert np.max(one_layer_training_split['train']['scaled_targets']) == 1.0
    assert np.max(two_layer_training_split['train']['scaled_targets']) == 1.0

    one_layer_shapes = get_shapes(one_layer_training_split, chunk_size=1000)
    two_layer_shapes = get_shapes(two_layer_training_split, chunk_size=1000)

    layers_dict_split = {
        1: one_layer_training_split,
        2: two_layer_training_split,
    }
    shapes = {
        1: one_layer_shapes,
        2: two_layer_shapes,
    }

    for path, (layer, layer_dict) in zip(savedirs, layers_dict_split.items()):
        for division, data in layer_dict.items():
            file = os.path.normpath(os.path.join(path, '{}.h5'.format(division)))
            if not os.path.exists(file):
                with h5py.File(file, 'w') as base_f:
                    print('[1/2] Created file:', file)
                    for data_type, values in data.items():
                        base_f.create_dataset(data_type, data=values, chunks=shapes[layer][data_type])

                with h5py.File(file, 'a') as modified_f:
                    print('[2/2] Now generating images...', '\n')
                    images = modified_f.create_dataset('images', (len(modified_f['input']),300,300,1), chunks=(1000,300,300,1))
                    for i, sample in enumerate(modified_f['input']):
                        img = image_process(sample)
                        images[i] = img

def scale_inputs(dictionary):
    for division in dictionary.keys():
        dictionary[division]['scaled_input'] = sample_scale(dictionary[division]['input'])

def scale_targets(dictionary):
    output_scaler = None
    for division in dictionary.keys():
        if output_scaler is None:
            output_scaler, scaled_target = output_scale(dictionary[division]['targets'], fit=True)
        else:
            output_scaler, scaled_target = output_scale(dictionary[division]['targets'], fit=False, scaler=output_scaler)
        dictionary[division]['scaled_targets'] = scaled_target
    
    return output_scaler

def get_shapes(dictionary, chunk_size=1000):
    shapes = {}
    for data_type, data in dictionary['train'].items():
        shapes[data_type] = (chunk_size, *data.shape[1:])

    return shapes 

def load_simulated_files_with_classes(files, no_layers):
    ''' Given list of .h5 files, load and concat into Numpy array, with classes '''
    y = None 
    x = None

    for f in files:
        if f.find('.h5') != -1:
            # print(f)
            with h5py.File(f, 'r') as file:
                x_i = np.squeeze(np.array(file.get('DATA')))
                y_i = np.array(file.get('SLD_NUMS'))
                # print(depth_i.shape)
                # c_i = classer(y_i)

                if (y is not None) & (x is not None):
                    x = np.concatenate((x, x_i), axis=0) 
                    y = np.concatenate((y, y_i), axis=0)
                    # c = np.concatenate((c, c_i), axis=0) 
                else:
                    x = x_i 
                    y = y_i
                    # c = c_i 

    c = np.full((len(y),1), no_layers)
    return {'input': x, 'targets': y, 'classes': c}

def splitter(data, split_ratios):
    random.seed(1)
    seed(1)
    length = len(data['input'])
    selected_dict = {}

    for division, ratio in split_ratios.items():
        random_sample = random.sample(range(0, length), int(length * ratio))
        selected_dict[division] = {}
        
        for name, values in data.items():
            selected_dict[division][name] = values[random_sample]
            values = np.delete(values, random_sample, 0)

    return selected_dict

def shuffler(dictionary):
    ''' Shuffled data, such that x, y, and class retains the same index. '''
    for training_split, data in dictionary.items():
        shuffled = np.arange(0, len(data['input']), 1)
        np.random.shuffle(shuffled)

        for name in data.keys():
            dictionary[training_split][name] = dictionary[training_split][name][shuffled]

def output_scale(t, fit=True, scaler=None):
    """Scale output values such that each has a min/max of 0/1. e.g. max: 1,1,1,1
    Arguments:
    t - np.array to scale
    fit - create and fit a new scaler
    scaler - predefined scaler (already fit)"""
    if fit == True:
        scaler = MinMaxScaler()
        scaler.fit(t)
    trans_t = scaler.transform(t)
        
    return (scaler, trans_t)

def sample_scale(x):
    """Scales both X and Y values in a sample using MinMaxScaling. Expects a 2D array as input"""
    ### for two dimensional data
    scaler = MinMaxScaler()
    x_scaled = []
    for sample in x:
        scaled = scaler.fit_transform(sample[:,1].reshape(-1,1))
        qs = scaler.fit_transform(sample[:,0].reshape(-1,1))
        x_scaled.append(np.array(list(zip(qs,scaled))))
    return np.array(x_scaled)

def get_image(x, y):
    """ Plot image using matplotlib and return as array """
    fig = plt.figure(figsize=(3,3))
    plt.plot(x,y)
    plt.yscale("log", basey=10)
    plt.xlim(0,0.3)
    plt.ylim(10e-8,1.5)
    plt.axis("off")
    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    mplimage = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    gray_image = color.rgb2gray(mplimage)
    plt.close()
    return gray_image

def image_process(sample):
    """ Return resized np array of image size (300,300,1) """
    x = sample[:,0]
    y = sample[:,1]
    image = get_image(x,y)
    return(np.resize(image, (300,300,1)))

if __name__ == "__main__":
    main()