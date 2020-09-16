import h5py
import os
import numpy as np

LAYERS_STR = {1: "one", 2: "two", 3: "three"}

def merge(save_path, layers_paths):
    save_path += "/merged"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for split in ['train', 'validate', 'test']:
        print(">>> Merging {}.h5 files".format(split))
        datasets = {dataset: [] for dataset in ('images', 'inputs', 'layers', 'targets')}
        
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
    layers = [1, 2]
    layers_paths = ["./models/investigate/test/data/{}".format(LAYERS_STR[layer]) for layer in layers]
    save_path = "./models/investigate/test/data"
    
    merge(save_path, layers_paths)
        