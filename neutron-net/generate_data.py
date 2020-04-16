import os
import sys
import glob
import h5py
import pickle
import random
import argparse

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from numpy.random import seed
from sklearn.preprocessing import MinMaxScaler
from skimage import data, color

def main(args):
    savedirs = [r'D:\Users\Public\Documents\stfc\neutron-net\data\test\1',
                 r'D:\Users\Public\Documents\stfc\neutron-net\data\test\2']

    savepath = r'D:\Users\Public\Documents\stfc\neutron-net\data\test'

    dat_files_dir = args.data

    one_layer_files = glob.glob(os.path.join(dat_files_dir, 'O') + '*')
    two_layer_files = glob.glob(os.path.join(dat_files_dir, 'Tw') + '*')

    if (not one_layer_files) or (not two_layer_files):
        print("\n   .dat files not found. Check data path.")
        return None
    else:
        print("\n   {} one-layer .dat file(s) found".format(len(one_layer_files)))
        print("   {} two-layer .dat file(s) found".format(len(two_layer_files)))
    
    layers_dict = {
        1: load_simulated_files(one_layer_files, 1),
        2: load_simulated_files(two_layer_files, 2),
    }

    # Check number of layers against size of array; trim if necessary
    for layers, data in layers_dict.items():
        length = data['targets'].shape[1]
        difference = length - layers * 2

        if difference:
            data['targets'] = data['targets'][:,:-difference]
    
    split_ratios = {'train': 0.8, 'validate': 0.1, 'test': 0.1}

    scaler_filename = os.path.join(savepath, 'output_scaler_{}-layer')
    
    split_layers_dict = train_valid_test_split(layers_dict, split_ratios)
    shuffle_data(split_layers_dict)
    scalers = scale_targets(split_layers_dict)
    dump_scalers(scalers, scaler_filename)
    scale_inputs(split_layers_dict) 
    shapes = get_shapes(split_layers_dict, chunk_size=1000)

    for no_layers, data in split_layers_dict.items():
        assert np.max(data['train']['scaled_targets']) == 1.0

    # for key, item in split_layers_dict.items():
    #     print('\n', '###', key)
    #     for key, item in item.items():
    #         print('----', key)
    #         for key, item in item.items():
    #             print('  ->', key)

    # for key, item in shapes.items():
    #     print(key, item)


    # for no_layers, layer_dictionary in layers_dict_split.items():
    #     print('###  Processing the {}-layer .dat files'.format(no_layers))

    #     for data_split_section_title, data_split in layer_dictionary.items():
    #         print('     -> Section:', data_split_section_title)
    #         file = os.path.normpath(os.path.join(savepath, '{}.h5'.format(data_split_section_title)))
    #         print('\n', file)
            
    # for path, (layer, layer_dict) in zip(savedirs, layers_dict_split.items()):
    #     print('### Layer: {}'.format(layer))

    #     for division, data in layer_dict.items():
    #         print('### Division: {}'.format(division))
    #         file = os.path.normpath(os.path.join(path, '{}.h5'.format(division)))

    #         if not os.path.exists(file):
    #             with h5py.File(file, 'w') as base_f:
    #                 print('[1/2] Created file:', file)
    #                 for data_type, values in data.items():
    #                     print('### Filling in data for {}'.format(data_type))
    #                     # base_f.create_dataset(data_type, data=values, chunks=shapes[layer][data_type])

    #             with h5py.File(file, 'a') as modified_f:
    #                 print('[2/2] Now generating images...', '\n')
    #                 # images = modified_f.create_dataset('images', (len(modified_f['input']),300,300,1), chunks=(1000,300,300,1))
    #                 # for i, sample in enumerate(modified_f['input']):
    #                 #     img = image_process(sample)
    #                 #     images[i] = img

def dump_scalers(scalers, scaler_filename):
    for no_layers, scaler in scalers.items():
        with open(scaler_filename.format(no_layers), 'wb') as f:
            pickle.dump(scaler, f)

def scale_inputs(split_layers_dict):
    for data in split_layers_dict.values():
        for division in data.keys():
            data[division]['scaled_inputs'] = sample_scale(data[division]['inputs'])

def scale_targets(split_layers_dict):
    scalers = {}

    for no_layers, dictionary in split_layers_dict.items(): 
        output_scaler = None   

        for division in dictionary.keys():
            if output_scaler is None:
                output_scaler, scaled_target = output_scale(dictionary[division]['targets'], fit=True)
            else:
                output_scaler, scaled_target = output_scale(dictionary[division]['targets'], fit=False, scaler=output_scaler)
            dictionary[division]['scaled_targets'] = scaled_target
        scalers[no_layers] = output_scaler

    return scalers

def get_shapes(split_layers_dict, chunk_size=1000):
    big_shapes = {}

    for no_layers, data in split_layers_dict.items():
        shapes = {}
        for data_type, data in data['train'].items():
            shapes[data_type] = (chunk_size, *data.shape[1:])
        big_shapes[no_layers] = shapes
    
    return big_shapes

def load_simulated_files(files, no_layers):
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
    return {'inputs': x, 'targets': y, 'layers': c}

def train_valid_test_split(layers_dict, split_ratios):
    split_layers_dict = {}
    for no_layers, data in layers_dict.items():
        random.seed(1)
        seed(1)
        length = len(data['inputs'])
        selected_dict = {}

        for division, ratio in split_ratios.items():
            random_sample = random.sample(range(0, length), int(length * ratio))
            selected_dict[division] = {}
            
            for name, values in data.items():
                selected_dict[division][name] = values[random_sample]
                values = np.delete(values, random_sample, 0)
             
        split_layers_dict[no_layers] = selected_dict
    return split_layers_dict

def shuffle_data(split_layers_dict):
    ''' Shuffled data, such that x, y, and class retains the same index. '''
    for dictionary in split_layers_dict.values():
        for training_split, data in dictionary.items():
            shuffled = np.arange(0, len(data['inputs']), 1)
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

def parse():
    parser = argparse.ArgumentParser(description='Data Generation')
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')  
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse()
    main(args)