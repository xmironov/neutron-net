import os, glob, h5py, random
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import seed
from skimage import color

LAYERS_STR   = {1: "one", 2: "two", 3: "three"}
DEPTH_BOUNDS = (20, 3000)
SLD_BOUNDS   = (-0.5, 10)

class ImageGenerator: 
    @staticmethod
    def scale_targets(concatenated):
        for split, data in concatenated.items():
            scaled_targets = np.zeros(data['targets'].shape)
            for i in range(3):
                scaled_targets[:, 2*i]   = ImageGenerator.scale_to_range(data['targets'][:, 2*i],   DEPTH_BOUNDS, (0, 1))
                scaled_targets[:, 2*i+1] = ImageGenerator.scale_to_range(data['targets'][:, 2*i+1], SLD_BOUNDS,   (0, 1))
            concatenated[split]['targets_scaled'] = scaled_targets
    
    @staticmethod
    def scale_to_range(old_value, old_range, new_range):
        old_min, old_max = old_range
        new_min, new_max = new_range
        return (((old_value - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min
    
    @staticmethod
    def get_shapes(concatenated, chunk_size=1000):
        shapes = {}
    
        for data_type, data in concatenated['train'].items():
            shapes[data_type] = (chunk_size, *data.shape[1:])
        
        return shapes
    
    @staticmethod
    def load_simulated_files(files, no_layers):
        ''' Given list of .h5 files, load and concat into Numpy array, with classes '''
        y = None 
        x = None
    
        for f in files:
            if f.find('.h5') != -1:
                with h5py.File(f, 'r') as file:
                    x_i = np.squeeze(np.array(file.get('DATA')))
                    y_i = np.array(file.get('SLD_NUMS'))
    
                    if (y is not None) & (x is not None):
                        x = np.concatenate((x, x_i), axis=0) 
                        y = np.concatenate((y, y_i), axis=0)
                    else:
                        x = x_i 
                        y = y_i
                        
        c = np.full((len(y), 1), no_layers)
        return {'inputs': x, 'targets': y, 'layers': c}
    
    @staticmethod
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
    
    @staticmethod
    def shuffle_data(split_layers_dict):
        ''' Shuffled data, such that x, y, and class retains the same index. '''
        for split in split_layers_dict.values():
            shuffled = np.arange(0, len(split['inputs']), 1)
            np.random.shuffle(shuffled)
    
            for type_of_data in split.keys():
                split[type_of_data] = split[type_of_data][shuffled]
    
    @staticmethod
    def image_process(sample):
        """ Return resized np array of image size (300,300,1) """
        x = sample[:,0]
        y = sample[:,1]
        image = ImageGenerator.__get_image(x,y)
        return(np.resize(image, (300,300,1)))
    
    @staticmethod
    def __get_image(x, y):
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

def generate_images(data_path, save_path, layers, chunk_size=1000, display_status=True):
    if any([layer <= 0 or layer > 3 for layer in layers]):
        print("Layers must be between 1 and 3 (inclusive)")
        return
    
    layers_files = {layer: glob.glob(os.path.join(data_path, LAYERS_STR[layer]) + '*') for layer in layers}
    layers_dict = {}

    i = 1
    for layer in layers:
        if layers_files[layer] == []:
            print("\n   {0}-layer .h5 file(s) not found. Check data path.".format(LAYERS_STR[layer]))
            return

        if display_status: print("\n   {0} {1}-layer .h5 file(s) selected".format(len(layers_files[layer]), LAYERS_STR[layer]))
        layers_dict[i] = ImageGenerator.load_simulated_files(layers_files[layer], layer)
        i += 1
  
    split_ratios = {'train': 0.8, 'validate': 0.1, 'test': 0.1}
    split_data = ImageGenerator.train_valid_test_split(layers_dict, split_ratios)
    
    concatenated = {}
    for split, layer_dict in split_data.items():
        concatenated[split] = {}

        for key in layer_dict[1].keys():
            concat = np.concatenate([layer_dict[layer][key] for layer in layer_dict.keys()])
            concatenated[split][key] = concat
    del split_data
   
    ImageGenerator.shuffle_data(concatenated)  #Shuffle data
    ImageGenerator.scale_targets(concatenated) #Scale targets

    shapes = ImageGenerator.get_shapes(concatenated, chunk_size=chunk_size)
    for section, dictionary in concatenated.items():
        file = os.path.normpath(os.path.join(save_path, '{}.h5'.format(section)))
        
        with h5py.File(file, 'w') as base_file:
            for type_of_data, data in dictionary.items():
                base_file.create_dataset(type_of_data, data=data, chunks=shapes[type_of_data])
        
        if display_status: 
            print("\n>>> Generating images for {}.h5".format(section))
        
        with h5py.File(file, 'a') as modified_file:
            images = modified_file.create_dataset('images', (len(modified_file['inputs']),300,300,1), chunks=(chunk_size,300,300,1))

            for i, sample in enumerate(modified_file['inputs']):
                img = ImageGenerator.image_process(sample)
                images[i] = img

if __name__ == "__main__":
    data_path = "./models/investigate/test/data/one"
    save_path = "./models/investigate/test/data/one"
    layers = [1]
    generate_images(data_path, save_path, layers, chunk_size=50)