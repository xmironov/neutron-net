import h5py
import os
import numpy as np

def merge(save_path, layers_paths):
    save_path += "/merge"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for split in ['train', 'validate', 'test']:
        datasets = {dataset: [] for dataset in ('images', 'inputs', 'inputs_scaled', 'layers', 'targets', 'targets_scaled')}
        
        for layer_path in layers_paths:
            with h5py.File(layer_path + "/{}.h5".format(split), 'r') as old_file:
                for dataset in datasets.keys():
                    datasets[dataset].append(np.array(old_file[dataset]))
            
        with h5py.File(save_path + "/{}.h5".format(split), 'w') as new_file:
            total_curves = len(datasets["layers"][1]) * len(layers_paths) # Number of curves per layer multiplied by the number of layers
            indices = np.arange(total_curves)
            np.random.shuffle(indices)

            for dataset in datasets.keys():
                concatenated = np.concatenate(datasets[dataset])
                new_file[dataset] = concatenated[indices]


if __name__ == "__main__":
    layers_str = {1: "One", 2: "Two", 3: "Three"}
    layers = [1, 2, 3]
    layers_paths = ["./models/investigate/classification/test/{}".format(layers_str[layer]) for layer in layers]
    save_path = "./models/investigate/classification/test"
    
    merge(save_path, layers_paths)
        