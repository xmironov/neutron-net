from comet_ml import Experiment

import os, time, re, glob, warnings
import argparse
import json
import h5py
import pickle

os.environ["KMP_AFFINITY"] = "none"

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.metrics import mean_squared_error, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, LeakyReLU, Dropout, Input
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Nadam

import confusion_matrix_pretty_print

DIMS = (300, 300)
CHANNELS = 1
tf.compat.v1.disable_eager_execution()

class Sequencer(Sequence):
    """ Use Keras Sequence class to load image data from h5 file"""
    def __init__(self, file, labels, dim, channels, batch_size, layers, debug=False, shuffle=False):
        self.indexes    = np.where(np.array(file["class"]) == layers)[0]
        self.file       = file
        self.labels     = labels
        self.dim        = dim
        self.channels   = channels
        self.batch_size = batch_size
        self.debug      = debug
        self.shuffle    = shuffle
        self.layers     = layers
        self.on_epoch_end()
    
    def __len__(self):
        """ Denotes number of batches per epoch"""
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, idx):
        """ Generates one batch of data"""
        indexes = self.indexes[idx*self.batch_size:(idx+1) * self.batch_size]
        inputs, targets = self.__data_generation(indexes)

        return inputs, targets

    def __data_generation(self, indexes):
        """ Generates data containing batch_size samples"""
        images = np.empty((self.batch_size, *self.dim, self.channels))
        targets_depth = np.empty((self.batch_size, self.layers), dtype=float)
        targets_sld = np.empty((self.batch_size, self.layers), dtype=float)

        for i, idx in enumerate(indexes):
            image = self.file["images"][idx]
            targets = self.labels[idx]
            length = len(targets)

            difference = length - self.layers * 2

            if difference:
                targets = targets[:-difference]

            images[i,] = np.array(image)
            targets_depth[i,] = targets[::2]    
            targets_sld[i,] = targets[1::2]

        return images, {"depth":targets_depth, "sld":targets_sld}
    
    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        indexes = np.where(np.array(self.file["class"]) == self.layers)[0]

        if self.shuffle:
            np.random.shuffle(indexes)

        self.indexes = indexes

class Regressor():
    def __init__(self, base, epochs, dropout, lr, workers, batch_size, layers):
        """ Initialisation"""
        self.base       = base
        self.epochs     = epochs
        self.dropout    = dropout
        self.lr         = lr
        self.workers    = workers
        self.batch_size = batch_size
        self.layers     = layers
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
            shuffle=False,
        )
        elapsed_time = time.time() - start
        self.time_taken = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        return self.history

    def test(self, test_sequence, test_labels, datapath, savepath, indexes):
        test_labels = test_labels[indexes]
        scaler = pickle.load(open(os.path.join(datapath, "output_scaler.p"), "rb"))
        predictions = self.model.predict(test_sequence, use_multiprocessing=False, verbose=1)
        depth, sld = predictions[0], predictions[1]
        
        if self.layers == 2:
            padded_depth = np.c_[depth[:,0], np.zeros(len(depth)), depth[:,1], np.zeros(len(depth))]
            padded_sld = np.c_[np.zeros(len(sld)), sld[:,0], np.zeros(len(sld)), sld[:,1]]
            transformed_depth = scaler.inverse_transorm(padded_depth)
            transformed_sld = scaler.inverse_transform(padded_sld)
            predictions = np.c_[transformed_depth[:,0], transformed_sld[:,1],
                                transformed_depth[:,2], transformed_sld[:,3]]

        elif self.layers == 1:
            padded_depth = np.c_[depth[:,0], np.zeros(len(depth)), np.zeros(len(depth)), np.zeros(len(depth))]
            padded_sld = np.c_[np.zeros(len(sld)), sld[:,0], np.zeros(len(sld)), np.zeros(len(sld))]
            transformed_depth = scaler.inverse_transform(padded_depth)
            transformed_sld = scaler.inverse_transform(padded_sld)
            predictions = np.c_[transformed_depth[:,0], transformed_sld[:,1],
                                np.zeros(len(sld)), np.zeros(len(sld))]

        # print("Labels length", len(test_labels), "Predictions length", len(predictions))
        remainder = len(test_labels) % self.batch_size

        if remainder:
            test_labels = test_labels[:-remainder]

        fig, ax = plt.subplots(2,2, figsize=(15,10))
        ax[0,0].scatter(test_labels[:,0], predictions[:,0], alpha=0.2)
        ax[0,0].set_title('Layer 1')
        ax[0,0].set_xlabel('Actual depth')
        ax[0,0].set_ylabel('Predicted depth')

        ax[0,1].scatter(test_labels[:,2], predictions[:,2], alpha=0.2)
        ax[0,1].set_title('Layer 2')
        ax[0,1].set_xlabel('Actual depth')
        ax[0,1].set_ylabel('Predicted depth')

        ax[1,0].scatter(test_labels[:,1], predictions[:,1], alpha=0.2)
        ax[1,0].set_xlabel('Actual SLD')
        ax[1,0].set_ylabel('Predicted SLD')

        ax[1,1].scatter(test_labels[:,3], predictions[:,3], alpha=0.2)
        ax[1,1].set_xlabel('Actual SLD')
        ax[1,1].set_ylabel('Predicted SLD')

        for i in range(2):
            ax[0,i].set_ylim(-100,3010)
            ax[0,i].set_xlim(-100,3010)
        
        for i in range(2):
            ax[1,i].set_ylim(-0.1,1.1)
            ax[1,i].set_xlim(-0.1,1.1)

        plt.savefig(os.path.join(savepath))

    def save(self, savepath):
        try:
            os.makedirs(savepath)
        except OSError:
            pass

        with open(os.path.join(savepath, "history.json"), "w") as f:
            json_dump = convert_to_float(self.history.history)
            json_dump["time_taken"] = self.time_taken
            json.dump(json_dump, f)

        model_yaml = self.model.to_yaml()

        with open(os.path.join(savepath, "model.yaml"), "w") as yaml_file:
            yaml_file.write(model_yaml)

        self.model.save_weights(os.path.join(savepath, "model_weights.h5"))

        with open(os.path.join(savepath, "summary.txt"), "w") as f:
            self.model.summary(print_fn=lambda x: f.write(x + "\n"))

        self.model.save(os.path.join(savepath, "full_model.h5"))

    def create_model(self):
        model = load_model(self.base)

        # for i in range(8):
        #     model.layers[i].trainable = False
        
        # for i in range(8, 24):
        #     model.layers[i].trainable = True
        for i in range(len(model.layers)):
            model.layers[i].trainable = True

        base_model = model.layers[26].output
        # depth_dense = Dense(50, activation="relu", name="depth_dense")(base_model)
        # sld_dense = Dense(50, activation="relu", name="sld_dense")(base_model)

        # dropout_depth = Dropout(self.dropout, name="dropout_depth")(depth_dense)
        # dropout_sld = Dropout(self.dropout, name="dropout_sld")(sld_dense)
        depth_output = Dense(units=self.layers, activation="linear", name="depth")(base_model)
        sld_output = Dense(units=self.layers, activation="linear", name="sld")(base_model)

        new_model = Model(inputs=model.input, outputs=[depth_output, sld_output])

        new_model.compile(
            loss={
                "depth":"mse",
                "sld":"mse",
            },
            loss_weights={
                "depth":1,
                "sld":1,
            },
            optimizer=Nadam(self.lr),
            metrics={
                "depth":"mae",
                "sld":"mae",
            }
        )
        return new_model
    
    def summary(self):
        self.model.summary()

def main(args):
    name = "regressor-%s-layer[" % str(args.layers) + datetime.now().strftime("%Y-%m-%dT%H%M%S") + "]"
    save = r"C:\Users\mtk57988\stfc\neutron-net\neutron-net\models\investigate"
    base = r"C:\Users\mtk57988\stfc\neutron-net\neutron-net\models\investigate\classifier-[2020-05-03T104626]\full_model.h5"
    data = r"D:\Users\Public\Documents\stfc\neutron-net\data\perfect-legacy\complete"

    savepath = os.path.join(save, name)

    if args.log:
        experiment = Experiment(api_key="Qeixq3cxlTfTRSfJ2hyPlMWjk", project_name="general", workspace="xandrovich")
    
    train_dir = os.path.join(data, "train.h5")
    validate_dir = os.path.join(data, "valid.h5")
    test_dir = os.path.join(data, "test.h5")

    train_file = h5py.File(train_dir, "r")
    validate_file = h5py.File(validate_dir, "r")
    test_file = h5py.File(test_dir, "r")

    targets = load_targets(data)
    train_targets, validate_targets, test_targets = targets["train"], targets["valid"], targets["test"]

    train_loader = Sequencer(train_file, train_targets, DIMS, CHANNELS, args.batch_size, args.layers)
    validate_loader = Sequencer(validate_file, validate_targets, DIMS, CHANNELS, args.batch_size, args.layers)
    test_loader = Sequencer(test_file, test_targets, DIMS, CHANNELS, args.batch_size, args.layers)

    # Define model
    model = Regressor(base, args.epochs, args.dropout_rate, args.learning_rate, args.workers, args.batch_size, args.layers)

    if args.summary:
        model.summary()

    indexes = np.where(np.array(test_file["class"]) == args.layers)[0]
    model.train(train_loader, validate_loader)
    model.test(test_loader, test_targets, data, savepath, indexes)

def parse():
    parser = argparse.ArgumentParser(description="Keras Regressor Training")

    # Meta Parameters
    parser.add_argument("data", metavar="PATH", help="path to data directory")
    parser.add_argument("save", metavar="PATH", help="path to save directory")
    parser.add_argument("base", metavar="PATH", help="path to base classifier model")
    parser.add_argument("layers", metavar="N", type=int, help="no. layers of system")
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

def load_targets(path):
    data = {}
    for section in ["train", "valid"]:
        with h5py.File(os.path.join(path, "{}.h5".format(section)), "r") as f:
            data["{}".format(section)] = np.array(f["scaledY"])

    with h5py.File(os.path.join(path, "test.h5"), "r") as f:
        data["test"] = np.array(f["Y"])

    return data

if __name__ == "__main__":
    args = parse()
    main(args)