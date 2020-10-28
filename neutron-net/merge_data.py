import h5py
import os, sys
import numpy as np
from generate_data import LAYERS_STR, DTYPES #String representations of each layer.

def merge(save_path, layers_paths, display_status=True):
    """Merges train, validate and test .h5 files for curves of different layers.
       This is used in training the classifier.

    Args:
        save_path (string): the file path to save the newly merged .h5 files to.
        layers_paths (List): a list of file paths to the directories of the files to merge.
        display_status (Boolean): whether to display merge progress.

    """
    save_path += "/merged" #Save under the "merged" directory.
    if not os.path.exists(save_path): #Make the directory if not already present.
        os.makedirs(save_path)

    for split in ('train', 'validate', 'test'): #Iterate over each split.
        print("\n>>> Merging {}.h5 files".format(split))

        #Load each of the old files for the split.
        old_files = [h5py.File(layer_path + "/{}.h5".format(split), 'r') for layer_path in layers_paths]
        #Check each file contains the same number of curves.
        for file in old_files[1:]:
            if len(file['layers']) != len(old_files[0]['layers']):
                sys.exit('All files must contain the same number of curves.')

        #Get the shapes of each dataset. This is used for defining the chunk size in the merged file.
        shapes = {dataset: old_files[0][dataset].shape[1:] for dataset in DTYPES.keys()}

        #Define the old and new number of curves.
        old_num_curves = old_files[0]['layers'].shape[0]
        new_num_curves = old_num_curves*len(old_files)

        with h5py.File(save_path + "/{}.h5".format(split), 'w') as new_file:
            old_chunk_size = 100
            new_chunk_size = old_chunk_size*len(old_files)

            #Create each new dataset for the new file.
            for dataset in shapes.keys():
                new_file.create_dataset(dataset, shape=(new_num_curves, *shapes[dataset]),
                                                 chunks=(new_chunk_size, *shapes[dataset]),
                                                 dtype=DTYPES[dataset])

            steps = int(new_num_curves / new_chunk_size) #Each step corresponds to writing a single chunk in the new file.
            for i in range(0, steps):
                new_start = i*new_chunk_size
                new_end   = (i+1)*new_chunk_size #Define the start and end of a chunk in the old files.

                old_start = i*old_chunk_size
                old_end   = (i+1)*old_chunk_size #Define the start and end of a chunk in the new file.

                indices = np.arange(new_chunk_size)
                np.random.shuffle(indices) #For shuffling the merged datasets.

                datasets = {dataset: [] for dataset in DTYPES.keys()}
                for dataset in datasets.keys():
                    for old_file in old_files:
                        datasets[dataset].append(np.array(old_file[dataset][old_start:old_end])) #Add the previous datasets to the new file.

                    concatenated = np.concatenate(datasets[dataset]) #Concatenate the list of separate datasets.
                    new_file[dataset][new_start:new_end] = concatenated[indices] #Randomly shuffle the new datasets

                if display_status and i % max(1, int(steps/10)) == 0:
                    print("   Writing chunk {0}/{1}...".format(i+int(steps/10), steps)) #Display how many chunks have been written so far.

        for old_file in old_files:
            old_file.close() #Close all the old files.


if __name__ == "__main__":
    layers = [1, 2, 3]
    layers_paths = ["./models/neutron/data/{}".format(LAYERS_STR[layer]) for layer in layers]
    save_path = "./models/neutron/data"

    merge(save_path, layers_paths, display_status=True)
