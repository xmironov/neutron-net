from comet_ml import Experiment

import os, time, re, glob, warnings
import argparse
import json
import h5py

os.environ["KMP_AFFINITY"] = "none"

import numpy as np 
import pandas as pd 

from datetime import datetime
from sklearn.metrics import mean_squared_error, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, LeakyReLU, Dropout, Input
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Nadam

import confusion_matrix_pretty_print

DIMS = (300, 300)
CHANNELS = 1
tf.compat.v1.disable_eager_execution()

class DataLoader(Sequence):
    """ Use Keras Sequence class to load image data from h5 file"""
    def __init__(self, file, labels, dim, channels, batch_size, debug=False, shuffle=False):
        self.file       = file
        self.labels     = labels
        self.dim        = dim
        self.channels   = channels
        self.batch_size = batch_size
        self.debug      = debug
        self.shuffle    = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        """ Denotes number of batches per epoch"""
        return int(np.floor(len(self.file["images"]) / self.batch_size))

    def __getitem__(self, idx):
        """ Generates one batch of data"""
        indexes = self.indexes[idx*self.batch_size:(idx+1) * self.batch_size]
        inputs, targets = self.__data_generation(indexes)

        return inputs, targets

    def __data_generation(self, indexes):
        """ Generates data containing batch_size samples"""
        images = np.empty((self.batch_size, *self.dim, self.channels))
        classes = np.empty((self.batch_size, 1), dtype=int)

        for i, idx in enumerate(indexes):
            image = self.file["images"][idx]
            images[i,] = np.array(image)
            classes[i,] = self.labels[idx]      
        return images, classes
    
    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        indexes = np.arange(len(self.file["images"]))

        if self.shuffle:
            np.random.shuffle(indexes)

        self.indexes = indexes

class Net():
    def __init__(self, dims, channels, epochs, dropout, lr, workers, batch_size):
        """ Initialisation"""
        self.dims       = dims
        self.channels   = channels
        self.epochs     = epochs
        self.dropout    = dropout
        self.lr         = lr
        self.workers    = workers
        self.batch_size = batch_size
        self.model      = self.create_model()

    def train(self, train_sequence, validate_sequence):
        """ Train and validate network"""
        learning_rate_reduction_cbk = ReduceLROnPlateau(
            monitor="val_loss",
            patience=10,
            verbose=1,
            factor=0.5,
            min_lr=0.000001,
        )

        start = time.time()
        self.history = self.model.fit(
            train_sequence,
            validation_data=validate_sequence,
            epochs=self.epochs,
            workers=self.workers,
            use_multiprocessing=False,
            verbose=1,
            callbacks=[learning_rate_reduction_cbk],
        )
        elapsed_time = time.time() - start
        self.time_taken = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        return self.history

    def test(self, test_sequence, test_labels, save_path):
        predictions = self.model.predict(test_sequence, use_multiprocessing=False, verbose=1)
        predictions = np.argmax(predictions, axis=1)
        remainder = len(test_labels) % self.batch_size

        if remainder:
            test_labels = test_labels[:-remainder]

        cm = confusion_matrix(test_labels, predictions)
        df_cm = pd.DataFrame(cm, index=[i for i in "12"], columns=[i for i in "12"])
        confusion_matrix_pretty_print.pretty_plot_confusion_matrix(df_cm, save_path)

    def save(self, save_path):
        try:
            os.makedirs(save_path)
        except OSError:
            print("Couldn't create savepath")

        with open(os.path.join(save_path, "history.json"), "w") as f:
            json_dump = convert_to_float(self.history.history)
            json_dump["time_taken"] = self.time_taken
            json.dump(json_dump, f)

        self.model.save(os.path.join(save_path, "full_model.h5"))

    def create_model(self):
        model = Sequential()

        model.add(Conv2D(32, (3,3), strides=(1,1), padding='same', activation="relu", input_shape=(*self.dims, self.channels)))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(64, (3,3), strides=(1,1), padding='same', activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(128, (3,3), strides=(1,1), padding='same', activation="relu"))
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))


        model.add(Conv2D(32, (3,3), strides=(1,1), padding='same', activation="relu"))
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

        model.add(Flatten()) # Length: 256 filters x 18 x 18 = 82944
        # model.add(Dense(150, activation="relu"))
        model.add(Dense(300, activation="relu"))
        model.add(Dropout(self.dropout))
        model.add(Dense(240, activation="relu"))
        model.add(Dropout(self.dropout))
        model.add(Dense(192, activation="relu"))
        model.add(Dropout(self.dropout))
        model.add(Dense(154, activation="relu"))
        model.add(Dropout(self.dropout))
        model.add(Dense(123, activation="relu"))
        model.add(Dropout(self.dropout))
        model.add(Dense(98, activation="relu"))
        model.add(Dropout(self.dropout))
        model.add(Dense(79, activation="relu"))
        model.add(Dropout(self.dropout))
        model.add(Dense(63, activation="relu"))
        model.add(Dropout(self.dropout))
        model.add(Dense(50, activation="relu"))
        model.add(Dropout(self.dropout))
        model.add(Dense(3, activation="softmax"))

        model.compile(
            optimizer=Nadam(self.lr),
            loss="sparse_categorical_crossentropy",
            metrics=["sparse_categorical_accuracy"]
        )
        return model
    
    def summary(self):
        self.model.summary()

def main(args):
    name = "classifier-[" + datetime.now().strftime("%Y-%m-%dT%H%M%S") + "]"
    save_path = os.path.join(args.save, name)

    if args.log:
        # Set up account with Comet-ML and retrieve api_key from them to track experiments
        # experiment = Experiment(api_key="", project_name="", workspace="")
        # experiment = Experiment(api_key="Qeixq3cxlTfTRSfJ2hyPlMWjk", project_name="general", workspace="xandrovich")

    train_dir = os.path.join(args.data, "train.h5")
    validate_dir = os.path.join(args.data, "valid.h5")
    test_dir = os.path.join(args.data, "test.h5")

    train_file = h5py.File(train_dir, "r")
    validate_file = h5py.File(validate_dir, "r")
    test_file = h5py.File(test_dir, "r")

    labels = load_labels(args.data)
    train_labels, validate_labels, test_labels = labels["train"], labels["valid"], labels["test"]

    train_loader = DataLoader(train_file, train_labels, DIMS, CHANNELS, args.batch_size, debug=False, shuffle=False)
    validate_loader = DataLoader(validate_file, validate_labels, DIMS, CHANNELS, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_file, test_labels, DIMS, CHANNELS, args.batch_size, shuffle=False)

    model = Net(DIMS, CHANNELS, args.epochs, args.dropout_rate, args.learning_rate, args.workers, args.batch_size)
    
    if args.summary:
        model.summary()

    model.train(train_loader, validate_loader)
    model.test(test_loader, test_labels, save_path)
    model.save(save_path)

    train_file.close()
    validate_file.close()
    test_file.close()

def parse():
    parser = argparse.ArgumentParser(description="Keras Classifier Training")
    # Meta Parameters
    parser.add_argument("data", metavar="PATH", help="path to data directory")
    parser.add_argument("save", metavar="PATH", help="path to save directory")
    parser.add_argument("-l", "--log", action="store_true", help="boolean: log metrics to CometML?")
    parser.add_argument("-s", "--summary", action="store_true", help="show model summary")

    # Model parameters
    parser.add_argument("-e", "--epochs", default=2, type=int, metavar="N", help="number of epochs")
    parser.add_argument("-b", "--batch_size", default=40, type=int, metavar="N", help="no. samples per batch (def:40)")
    parser.add_argument("-j", "--workers", default=1, type=int, metavar="N", help="no. data loading workers (def:1)")
    
    # Learning parameters
    parser.add_argument("-lr", "--learning_rate", default=0.0003, type=float, metavar="R", help="Nadam learning rate")
    parser.add_argument("-dr", "--dropout_rate", default=0.1, type=float, metavar="R", help="dropout rate" )
    return parser.parse_args()

def load_labels(path):
    data = {}
    for section in ["train", "valid", "test"]:
        with h5py.File(os.path.join(path, "{}.h5".format(section)), "r") as f:
            data["{}".format(section)] = np.array(f["class"])

    return data

def convert_to_float(dictionary):
	""" For saving model output to json"""
	jsoned_dict = {}
	for key in dictionary.keys():
		if type(dictionary[key]) == list:
			jsoned_dict[key] = [float(i) for i in dictionary[key]]
		else:
			jsoned_dict[key] = float(dictionary[key])
	return jsoned_dict

if __name__ == "__main__":
    args = parse()
    main(args)