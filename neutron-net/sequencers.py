import numpy as np

from tensorflow.keras.utils import Sequence

class DataSequence(Sequence):
    ''' Use Keras sequence to load image data from h5 file '''
    def __init__(self, dim, channels, batch_size, mode=None, layers=None, h5_file=None, labels=None):
        'Initialisation'
        # Potential Inputs
        self.file       = h5_file
        self.labels     = labels

        # Parameters
        self.dim        = dim                     # Image dimensions
        self.channels   = channels                # Image channels                   
        self.batch_size = batch_size              # Batch size
        self.mode       = mode
        self.layers     = layers
        self.on_epoch_end()

    def __len__(self):
        'Denotes number of batches per epoch'
        if self.file:
            return int(np.floor(len(self.file['images']) / self.batch_size))
        elif self.labels:
            return int(np.floor(len(self.labels.keys()) / self.batch_size))

    def __getitem__(self, index):
        'Generates one batch of data'
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        if self.labels:
            indexes = [list(self.labels.keys())[k] for k in indexes]
        images, values = self.__data_generation(indexes)

        return images, values

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'
        images = np.empty((self.batch_size, *self.dim, self.channels))

        if self.mode == 'regression':
            if self.file:
                targets_depth = np.empty((self.batch_size, self.layers), dtype=float)
                targets_sld = np.empty((self.batch_size, self.layers), dtype=float)
                for i, idx in enumerate(indexes):
                    image = self.file['images'][idx]
                    target = self.file['scaled_targets'][idx]
                    images[i,] = image
                    targets_depth[i,] = target[::2]
                    targets_sld[i,] = target[1::2]
                return images, {'depth': targets_depth, 'sld': targets_sld}
            
            elif self.labels:
                targets_depth = []
                targets_sld = []
                for i, np_image_filename in enumerate(indexes):
                    images[i,] = np.load(np_image_filename)
                    targets_depth.append(self.labels[np_image_filename]['depth'])
                    targets_sld.append(self.labels[np_image_filename]['sld'])
                return images, {'depth': np.array(targets_depth), 'sld': np.array(targets_sld)}

        if self.mode == 'classification':
            if self.file:
                classes = np.empty((self.batch_size, 1), dtype=int)
                for i, idx in enumerate(indexes):
                    image = self.file['images'][idx]
                    images[i,] = image
                    classes[i,] = self.file['classes'][idx]
                return images, classes

            elif self.labels:
                for i, np_image_filename in enumerate(indexes):
                    images[i,] = np.load(np_image_filename)
                    classes[i,] = self.labels[np_image_filename]
                return images, classes

        # for i, idx in enumerate(indexes):
        #     image = self.file['images'][idx]
        #     values = self.file['scaledY'][idx]
        #     length = len(values)
        #     difference = length - self.layers * 2

        #     if difference:
        #         # print(difference)
        #         values = values[:-difference]

        #     # fill preallocated arrays
        #     x[i,] = image
        #     y_depth[i,] = values[::2]
        #     y_sld[i,] = values[1::2]     

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.file:
            self.indexes = np.arange(len(self.file['images']))
        elif self.labels:
            self.indexes = np.arange(len(self.labels.keys()))

    def close_file(self):
        self.file.close()