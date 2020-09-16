import os, h5py
os.environ["KMP_AFFINITY"] = "none"
import numpy as np 

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.models     import Model, load_model
from tensorflow.keras.layers     import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Input
from tensorflow.keras.utils      import Sequence
from tensorflow.keras.callbacks  import ReduceLROnPlateau
from tensorflow.keras.optimizers import Nadam

DIMS = (300, 300)
CHANNELS = 1
LAYERS_STR = {1: "one", 2: "two", 3: "three"}

class DataLoader(Sequence):
    ''' Use Keras sequence to load image data from h5 file '''
    def __init__(self, h5_file, dim, channels, batch_size, layers):
        'Initialisation'
        self.file       = h5_file                 # H5 file to read
        self.dim        = dim                     # Image dimensions
        self.channels   = channels                # Image channels                   
        self.batch_size = batch_size              # Batch size
        self.layers     = layers
        self.on_epoch_end()

    def __len__(self):
        'Denotes number of batches per epoch'
        return int(np.floor(len(np.array(self.file['images'])) / self.batch_size))

    def __getitem__(self, index):
        'Generates one batch of data'
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        images, targets = self.__data_generation(indexes)

        return images, targets

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'
        images = np.empty((self.batch_size, *self.dim, self.channels))
        targets_depth = np.empty((self.batch_size, self.layers), dtype=float)
        targets_sld = np.empty((self.batch_size, self.layers), dtype=float)

        for i, idx in enumerate(indexes):
            image = self.file['images'][idx]
            values = self.file['targets_scaled'][idx]

            length = len(values)
            difference = length - self.layers * 2

            if difference:
                values = values[:-difference]

            images[i,] = image
            targets_depth[i,] = values[::2]
            targets_sld[i,] = values[1::2]
        
        return images, {'depth': targets_depth, 'sld': targets_sld}

    def on_epoch_end(self):
        'Updates indexes after each epoch'    
        self.indexes = np.arange(len(self.file['images']))

    def close_file(self):
        self.file.close()


class Regressor():
    def __init__(self, dims, channels, layers, epochs, learning_rate, batch_size, dropout, workers, load_path=None):
        'Initialisation'
        self.dims          = dims
        self.channels      = channels
        self.outputs       = layers
        self.epochs        = epochs
        self.learning_rate = learning_rate
        self.batch_size    = batch_size
        self.dropout       = dropout
        self.workers       = workers
        if load_path is None:
            self.model = self.create_model()
        else:
            self.model = load_model(load_path)

    def train(self, train_seq, valid_seq):
        'Trains data on Sequences'

        learning_rate_reduction_cbk = ReduceLROnPlateau(
            monitor='val_loss',
            patience=10,
            verbose=1,
            factor=0.5,
            min_lr = 0.000001
        )

        self.history = self.model.fit(
            train_seq,
            validation_data = valid_seq,
            epochs = self.epochs,
            workers = self.workers,
            use_multiprocessing = False,
            verbose = 1,
            callbacks = [learning_rate_reduction_cbk]
        )

        train_seq.close_file()
        valid_seq.close_file()
        return self.history
    
    def test(self, test_seq):
        print("Evaluating")
        results = self.model.evaluate(test_seq)
        print("Depth Loss: {0} | SLD Loss: {1}\nDepth mae:  {2} | SLD mae:  {3}".format(results[1], results[2], results[3], results[4]))

    def create_model(self):
        # Convolutional Encoder
        input_img = Input(shape=(*self.dims, self.channels))
        conv_1 = Conv2D(32, (3,3), activation='relu')(input_img)
        pool_1 = MaxPooling2D((2,2))(conv_1)
        conv_2 = Conv2D(64, (3,3), activation='relu')(pool_1)
        pool_2 = MaxPooling2D((2,2), strides=(2,2))(conv_2)
        conv_3 = Conv2D(32, (3,3), activation='relu')(pool_2)
        pool_3 = MaxPooling2D((2,2))(conv_3)
        conv_4 = Conv2D(16, (3,3), activation='relu')(pool_3)
        pool_4 = MaxPooling2D((2,2))(conv_4)
        flatten = Flatten()(pool_4)

        # Depth feed-forward
        dense_1_d = Dense(units=300, activation='relu', kernel_initializer='he_normal')(flatten)
        dropout_1_d = Dropout(self.dropout)(dense_1_d)
        dense_2_d = Dense(units=192, activation='relu', kernel_initializer='he_normal')(dropout_1_d)
        dropout_2_d = Dropout(self.dropout)(dense_2_d)
        dense_3_d = Dense(units=123, activation='relu', kernel_initializer='he_normal')(dropout_2_d)
        dropout_3_d = Dropout(self.dropout)(dense_3_d)
        dense_4_d = Dense(units=79, activation='relu', kernel_initializer='he_normal')(dropout_3_d)
        dropout_4_d = Dropout(self.dropout)(dense_4_d)
        dense_5_d = Dense(units=50, activation='relu', kernel_initializer='he_normal')(dropout_4_d)
        dropout_5_d = Dropout(self.dropout)(dense_5_d)
        depth_linear = Dense(units=self.outputs, activation='linear', name='depth')(dropout_5_d)
        sld_linear = Dense(units=self.outputs, activation='linear', name='sld')(dropout_5_d)

        model = Model(inputs=input_img, outputs=[depth_linear, sld_linear])
        model.compile(loss={'depth':'mse','sld':'mse'},
                        loss_weights={'depth':1,'sld':1},
                        optimizer = Nadam(self.learning_rate),
                        metrics={'depth':'mae','sld':'mae'})
        return model

    def save(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        self.model.save(os.path.join(save_path, 'full_model.h5'))

    def summary(self):
        self.model.summary()


def regress(data_path, layer, save_path=None, load_path=None, train=True, summary=False, epochs=2,
         learning_rate=0.0004, batch_size=20, dropout_rate=0.1, workers=1):
    if save_path is not None:
        save_path = os.path.join(save_path, "{}-layer-regressor".format(LAYERS_STR[layer]))
    
    train_dir = os.path.join(data_path, 'train.h5')
    val_dir   = os.path.join(data_path, 'validate.h5')
    test_dir  = os.path.join(data_path, 'test.h5')

    train_h5 = h5py.File(train_dir, 'r')
    val_h5   = h5py.File(val_dir, 'r')
    test_h5  = h5py.File(test_dir, 'r')

    train_loader = DataLoader(train_h5, DIMS, CHANNELS, batch_size, layer)
    valid_loader = DataLoader(val_h5,   DIMS, CHANNELS, batch_size, layer)
    test_loader  = DataLoader(test_h5,  DIMS, CHANNELS, batch_size, layer)

    model = Regressor(DIMS, CHANNELS, layer, epochs, learning_rate, batch_size, dropout_rate, workers, load_path)
    if summary:
        model.summary()

    if train:
        model.train(train_loader, valid_loader)
    model.test(test_loader) 
    if save_path is not None:
        model.save(save_path)

if __name__ == "__main__":
    layer     = 1
    data_path = "./models/investigate/test/data/{}".format(LAYERS_STR[layer])
    save_path = "./models/investigate/test"
    #load_path = "./models/investigate/test/{}-layer-regressor/full_model.h5".format(LAYERS_STR[layer])
    
    regress(data_path, layer, save_path, epochs=1)