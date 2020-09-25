import os, glob, h5py, random
import matplotlib.pyplot as plt
import numpy as np
from skimage import color

LAYERS_STR   = {1: "one", 2: "two", 3: "three"}

class ImageGenerator:
    """The ImageGenerator class generates images from reflectivity data.
    
    Class Attributes:
        depth_bounds (tuple): bounds on depth used for scaling targets.
        sld_bounds (tuple): bounds on sld used for scaling targets.
        
    """
    depth_bounds = (20, 3000)
    sld_bounds   = (-0.5, 10)

    @staticmethod
    def scale_targets(concatenated):
        """Scales target (SLD and depth) values to be in the range [0,1].

        Args:
            concatenated (Dataset): h5py dataset containing train, validate and test data.

        """
        for split, data in concatenated.items(): #Iterate over each split.
            scaled_targets = np.zeros(data['targets'].shape) #Blank array of zeros to fill in.
            for i in range(3): #Apply scaling to the depth and sld values for each layer.
                scaled_targets[:, 2*i]   = ImageGenerator.scale_to_range(data['targets'][:, 2*i],   ImageGenerator.depth_bounds, (0, 1))
                scaled_targets[:, 2*i+1] = ImageGenerator.scale_to_range(data['targets'][:, 2*i+1], ImageGenerator.sld_bounds,   (0, 1))
            concatenated[split]['targets_scaled'] = scaled_targets

    @staticmethod
    def scale_to_range(old_value, old_range, new_range):
        """Scales a given `old_value` in an `old_range` to a new value in the `new_range`.

        Args:
            old_value (ndarray): an array of values to scale.
            old_range (tuple): a range encompassing the values of `old_value`
            new_range (tuple): the range to scale values to.

        Returns:
            The scaled result in `new_range`.

        """
        old_min, old_max = old_range
        new_min, new_max = new_range
        return (((old_value - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min

    @staticmethod
    def get_shapes(concatenated, chunk_size=1000):
        """Retrieves the shape of each component of the given dataset.

        Args:
            concatenated (Dataset): h5py dataset containing train, validate and test data.
            chunk_size (type): the size of chunks to create the final h5 file with.

        Returns:
            A dictionary shapes in the form of tuples corresponding to each data type.

        """
        shapes = {}
        for data_type, data in concatenated['train'].items():
            shapes[data_type] = (chunk_size, *data.shape[1:])
        return shapes

    @staticmethod
    def load_simulated_files(files, no_layers):
        """Given list of .h5 files, loads and concatenates into Numpy array, with classes.

        Args:
            files (list): list of files to load from.
            no_layers (int): the number of layers the data in the file has.

        Returns:
            A dictionary of inputs, targets and classes for the given files.

        """
        y = None
        x = None
        for f in files: #Iterate over each file and try to find it.
            if f.find('.h5') != -1:
                with h5py.File(f, 'r') as file:
                    x_i = np.squeeze(np.array(file.get('DATA')))
                    y_i = np.array(file.get('SLD_NUMS'))

                    #Concatenate data of the file with the existing loaded data.
                    if (y is not None) & (x is not None):
                        x = np.concatenate((x, x_i), axis=0)
                        y = np.concatenate((y, y_i), axis=0)
                    else:
                        x = x_i
                        y = y_i

        #The class for each curve is the number of layers it was generated with.
        c = np.full((len(y), 1), no_layers)
        return {'inputs': x, 'targets': y, 'layers': c}

    @staticmethod
    def train_valid_test_split(layers_dict, split_ratios):
        """Splits data into training, validate and test splits.

        Args:
            layers_dict (dict): the data for each layer to split.
            split_ratios (dict): the proportion of data for each split.

        Returns:
            A dictionary with the original layer data split by the given ratios.

        """
        split_layers_dict = {}
        for split, split_value in split_ratios.items():
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

    @staticmethod
    def shuffle_data(split_layers_dict):
        """Shuffles data, such that x, y, and class retain the same index.

        Args:
            split_layers_dict (dict): the reflectivity data for each split.

        """
        for split in split_layers_dict.values(): #Iterate over each split.
            indices = np.arange(0, len(split['inputs']), 1)
            np.random.shuffle(indices) #Shuffle the indices

            for type_of_data in split.keys():
                split[type_of_data] = split[type_of_data][indices]

    @staticmethod
    def image_process(sample):
        """Processes a sample by generating an image for it and resizing it.

        Args:
            sample (ndarray): the sample to convert to an image.

        Returns:
            np array (image) of size (300,300,1).

        """
        q = sample[:,0]
        r = sample[:,1]
        image = ImageGenerator.__get_image(q, r)
        return(np.resize(image, (300, 300, 1)))

    @staticmethod
    def __get_image(q, r):
        """Plots image using matplotlib and returns as an array.

        Args:
            q (ndarray): an array of momentum transfer values.
            r (ndarray): an array of reflectance values.

        Returns:
            np array corresponding to an image of the original reflectivity data.

        """
        fig = plt.figure(figsize=(3,3)) #Create matplotlib figure and setup axes.
        plt.plot(q, r)
        plt.yscale("log")
        plt.xlim(0, 0.3)
        plt.ylim(10e-8, 1.5)
        plt.axis("off")
        fig.canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi() #Resize
        mplimage = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        gray_image = color.rgb2gray(mplimage) #Convert to grey scale.
        plt.close()
        return gray_image

def generate_images(data_path, save_path, layers, chunk_size=1000, display_status=True):
    """Generates images for reflectivity data of varying layers in a given `data_path`.

    Args:
        data_path (type): the directory containing data to convert to images.
        save_path (type): the directory to store the newly generated images to.
        layers (type): the layers for which the corresponding files are to be converted.
        chunk_size (type): the size of chunks used when storing images in a .h5 file.
        display_status (type): whether to display the image generation progress.

    """
    if any([layer <= 0 or layer > 3 for layer in layers]): #Check layers are valid
        print("Layers must be between 1 and 3 (inclusive)")
        return

    #Get the paths of each file for each number of layers and store as a dictionary.
    layers_files = {layer: glob.glob(os.path.join(data_path, LAYERS_STR[layer]) + '*') for layer in layers}
    layers_dict = {}

    i = 1
    for layer in layers:
        if layers_files[layer] == []: #Check that files can be found for the given number of layers.
            print("\n   {0}-layer .h5 file(s) not found. Check data path.".format(LAYERS_STR[layer]))
            return

        if display_status:
             print("\n   {0} {1}-layer .h5 file(s) selected".format(len(layers_files[layer]), LAYERS_STR[layer]))
        layers_dict[i] = ImageGenerator.load_simulated_files(layers_files[layer], layer) #Load the found file.
        i += 1

    split_ratios = {'train': 0.8, 'validate': 0.1, 'test': 0.1} #Split data into train, validate and test.
    split_data = ImageGenerator.train_valid_test_split(layers_dict, split_ratios)

    concatenated = {}
    for split, layer_dict in split_data.items():
        concatenated[split] = {}

        for key in layer_dict[1].keys():
            concat = np.concatenate([layer_dict[layer][key] for layer in layer_dict.keys()])
            concatenated[split][key] = concat
    del split_data

    ImageGenerator.shuffle_data(concatenated)  #Shuffle concatenated data
    ImageGenerator.scale_targets(concatenated) #Scale the targets to be between 0 and 1.

    shapes = ImageGenerator.get_shapes(concatenated, chunk_size=chunk_size)
    for section, dictionary in concatenated.items():
        file = os.path.normpath(os.path.join(save_path, '{}.h5'.format(section)))

        with h5py.File(file, 'w') as base_file: #Create the file for the current split.
            for type_of_data, data in dictionary.items():
                base_file.create_dataset(type_of_data, data=data, chunks=shapes[type_of_data])

        if display_status:
            print("\n>>> Generating images for {}.h5".format(section))

        with h5py.File(file, 'a') as modified_file:
            images = modified_file.create_dataset('images', (len(modified_file['inputs']),300,300,1), chunks=(chunk_size,300,300,1))

            for i, sample in enumerate(modified_file['inputs']): #Create images for each sample.
                img = ImageGenerator.image_process(sample)
                images[i] = img

if __name__ == "__main__":
    data_path = "./models/investigate/data/two"
    save_path = "./models/investigate/data/two"
    layers = [2]
    generate_images(data_path, save_path, layers, chunk_size=500)
