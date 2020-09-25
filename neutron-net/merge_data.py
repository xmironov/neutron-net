import h5py
import os
import numpy as np
from generate_data import LAYERS_STR #String representations of each layer.

def merge(save_path, layers_paths):
    """Merges train, validate and test .h5 files for curves of different layers.
       This is used in training the classifier.

    Args:
        save_path (String): the file path to save the newly merged .h5 files to.
        layers_paths (List): a list of file paths to the directories of the files to merge.

    """
    save_path += "/merged" #Save under the "merged" directory.
    if not os.path.exists(save_path): #Make the directory if not already present.
        os.makedirs(save_path)

    for split in ['train', 'validate', 'test']: #Iterate over each split.
        print(">>> Merging {}.h5 files".format(split))
        datasets = {dataset: [] for dataset in ('images', 'inputs', 'layers', 'targets', 'targets_scaled')}

        for layer_path in layers_paths: #Iterate over each directory for each of the layers' data to merge.
            with h5py.File(layer_path + "/{}.h5".format(split), 'r') as old_file:
                for dataset in datasets.keys():
                    datasets[dataset].append(np.array(old_file[dataset])) #Add the previous datasets to the new file.

        with h5py.File(save_path + "/{}.h5".format(split), 'w') as new_file:
            total_curves = len(datasets["layers"][1]) * len(layers_paths) # Number of curves per layer multiplied by the number of layers
            indices = np.arange(total_curves)
            np.random.shuffle(indices)

            for dataset in datasets.keys():
                concatenated = np.concatenate(datasets[dataset]) #Concatenate the list of separate datasets.
                new_file[dataset] = concatenated[indices] #Randomly shuffle the new datasets


if __name__ == "__main__":
    layers = [1, 2]
    layers_paths = ["./models/investigate/data/{}".format(LAYERS_STR[layer]) for layer in layers]
    save_path = "./models/investigate/data"

    merge(save_path, layers_paths)
