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
        return int(np.floor(len(self.file['images']) / self.batch_size))

    def __getitem__(self, index):
        'Generates one batch of data'
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
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

        if self.mode == 'classification':
            if self.file:
                classes = np.empty((self.batch_size, 1), dtype=int)

                for i, idx in enumerate(indexes):
                    image = self.file['images'][idx]

                    images[i,] = image
                    classes[i,] = self.file['classes'][idx]

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
        self.indexes = np.arange(len(self.file['images']))

    def close_file(self):
        self.file.close()

class DataSequenceClasses(Sequence):
    ''' Use Keras sequence to load image data from h5 file '''
    def __init__(self, labels, dim, channels, batch_size):
        'Initialisation'
        self.labels     = labels
        self.dim        = dim                     # Image dimensions
        self.channels   = channels                # Image channels                   
        self.batch_size = batch_size              # Batch size
        self.on_epoch_end()

    def __len__(self):
        'Denotes number of batches per epoch'
        return int(np.floor(len(self.labels.keys()) / self.batch_size))

    def __getitem__(self, index):
        'Generates one batch of data'
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        np_image_batch = [list(self.labels.keys())[k] for k in indexes]
        x, c = self.__data_generation(np_image_batch)

        return x, c

    def __data_generation(self, np_image_batch):
        'Generates data containing batch_size samples'
        x = np.empty((self.batch_size, *self.dim, self.channels))
        c = np.empty((self.batch_size, 1), dtype=int)

        for i, np_image in enumerate(np_image_batch):
            x[i,] = np.load(np_image)
            c[i,] = self.labels[np_image]
        
        return x, c        

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.labels.keys()))

    def close_file(self):
        self.file.close()