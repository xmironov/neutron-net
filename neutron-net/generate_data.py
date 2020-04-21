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
    
    split_ratios = {'train': 0.8, 'validate': 0.1, 'test': 0.1}
    scaler_filename = os.path.join(savepath, 'output_scaler.p')
    split_data = train_valid_test_split(layers_dict, split_ratios)
    
    concatenated = {}
    for split, data in split_data.items():
        concatenated[split] = {}

        for key in data[1].keys():
            concat = np.concatenate([data[layer][key] for layer in data.keys()])
            concatenated[split][key] = concat
    del split_data
              
    print('\n### Shuffling data')
    shuffle_data(concatenated)
    print('\n### Scaling targets')
    output_scaler = scale_targets(concatenated)
    with open(scaler_filename, 'wb') as f:
            pickle.dump(output_scaler, f)

    print('\n### Scaling inputs')
    scale_inputs(concatenated) 
    shapes = get_shapes(concatenated, chunk_size=1000)
    print('\n### Creating .h5 files')
    for section, dictionary in concatenated.items():
        file = os.path.normpath(os.path.join(savepath, '{}.h5'.format(section)))
        
        if not os.path.exists(file):
            print('\n### Filling in data for {}.h5'.format(section))
            with h5py.File(file, 'w') as base_file:
                for type_of_data, data in dictionary.items():
                    base_file.create_dataset(type_of_data, data=data, chunks=shapes[type_of_data])
            
            print('\n### Generating images for {}.h5'.format(section))
            with h5py.File(file, 'a') as modified_file:
                images = modified_file.create_dataset('images', (len(modified_file['inputs']),300,300,1), chunks=(1000,300,300,1))

                for i, sample in enumerate(modified_file['inputs']):
                    img = image_process(sample)
                    images[i] = img

def scale_inputs(concatenated):
    for regime, data in concatenated.items():
        concatenated[regime]['inputs_scaled'] = sample_scale(data['inputs'])

    # for data in split_data.values():
    #     for layer in data.keys():
    #         data[layer]['scaled_inputs'] = sample_scale(data[layer]['inputs'])

def scale_targets(concatenated):
    output_scaler = None

    for regime, data in concatenated.items():
        if output_scaler is None:
            output_scaler, scaled_target = output_scale(data['targets'], fit=True)
            assert np.max(scaled_target) == 1.0
            
        elif output_scaler:
            output_scaler, scaled_target = output_scale(data['targets'], fit=False, scaler=output_scaler)
            assert np.max(scaled_target) == 1.0
        
        concatenated[regime]['targets_scaled'] = scaled_target

    return output_scaler

def get_shapes(concatenated, chunk_size=1000):
    shapes = {}

    for data_type, data in concatenated['train'].items():
        shapes[data_type] = (chunk_size, *data.shape[1:])
    
    return shapes

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

                if (y is not None) & (x is not None):
                    x = np.concatenate((x, x_i), axis=0) 
                    y = np.concatenate((y, y_i), axis=0)
                else:
                    x = x_i 
                    y = y_i
    c = np.full((len(y),1), no_layers)

    return {'inputs': x, 'targets': y, 'layers': c}

def train_valid_test_split(layers_dict, split_ratios):
    split_layers_dict = {}
    # layers_dict = {1: {'inputs':..,'targets':..,}, 2:{..}}

    for split, split_value in split_ratios.items():
        random.seed(1)
        seed(1)
        selected_dict = {}

        for no_layers, layer_dict in layers_dict.items():
            length = len(layer_dict['inputs'])
            random_sample = random.sample(range(0, length), int(length * split_value))
            selected_dict[no_layers] = {}

            for data_type, data in layer_dict.items():
                selected_dict[no_layers][data_type] = data[random_sample]
                data = np.delete(data, random_sample, axis=0)
        
        split_layers_dict[split] = selected_dict

    return split_layers_dict


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
    for split in split_layers_dict.values():
        shuffled = np.arange(0, len(split['inputs']), 1)
        np.random.shuffle(shuffled)

        for type_of_data in split.keys():
            split[type_of_data] = split[type_of_data][shuffled]

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

# test

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
    parser.add_argument('data', metavar='DATADIR',
                        help='path to dataset')  
    parser.add_argument('save', metavar='SAVEDIR',
                        help='path to save directory')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse()
    main(args)