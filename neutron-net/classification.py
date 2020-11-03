import h5py, os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Nadam
tf.compat.v1.disable_eager_execution()

from generate_data import DIMS, CHANNELS, IMAGE_BITS
from confusion_matrix_pretty_print import ConfusionMatrixPrinter

class DataLoader(Sequence):
    """DataLoader uses a Keras Sequence to load image data from a h5 file."""

    def __init__(self, file, labels, dim, channels, batch_size, shuffle=False):
        """Initialises the DataLoader class with given parameters.

        Args:
            file (string): the path of the file to load data from.
            labels (ndarray): the labels corresponding to the loaded data.
            dim (tuple): dimensions of images loaded.
            channels (int): number of channels of images loaded.
            batch_size (int): size of each mini-batch to load.
            shuffle (Boolean): whether to shuffle loaded data or not.

        """
        self.file       = file
        self.labels     = labels
        self.dim        = dim
        self.channels   = channels
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.__on_epoch_end()

    def __len__(self):
        """Calculates the number of batches per epoch.

        Returns:
            An integer number of batches per epoch.

        """
        return int(np.floor(len(self.file["images"]) / self.batch_size))

    def __getitem__(self, index):
        """Generates one batch of data.

        Args:
            index (int): position of batch.

        Returns:
            A `batch_size` sample of images (inputs) and classes (targets).

        """
        indices = self.indices[index*self.batch_size: (index+1)*self.batch_size] #Get range of indices.
        inputs, targets = self.__data_generation(indices)
        return inputs, targets

    def __data_generation(self, indices):
        """Generates data containing batch_size samples.

        Args:
            indices (ndarray): an array of indices to retrieve data from.

        Returns:
            A `batch_size` sample of images (inputs) and classes (targets).

        """
        images  = np.empty((self.batch_size, *self.dim, self.channels))
        classes = np.empty((self.batch_size, 1), dtype=int)

        for i, idx in enumerate(indices): #Get images and classes for each index
            images[i,]  = np.array(self.file["images"][idx]) / (2**IMAGE_BITS) #Divide to get images back into 0-1 range.
            classes[i,] = self.labels[idx]

        return images, classes

    def __on_epoch_end(self):
        """Updates indices after each epoch."""
        indices = np.arange(len(self.file["images"]))
        if self.shuffle:
            np.random.shuffle(indices)
        self.indices = indices


class Classifier():
    """The Classifier class represents the network used for layer classification."""

    def __init__(self, dims, channels, epochs, lr, batch_size, dropout, workers, load_path=None):
        """Initialises the network with given hyperparameters.

        Args:
            dims (tuple): dimensions of the input images.
            channels (int): number of channels of the input images.
            epochs (int): number of epochs to train for.
            lr (float): the value for the learning rate hyperparameter.
            batch_size (int): the size of each mini-batch.
            dropout (float): the value of the dropout rate hyperparameter.
            workers (int): number of workers to use.
            load_path (string): the path an existing model to load from.

        """
        self.dims       = dims
        self.channels   = channels
        self.epochs     = epochs
        self.lr         = lr
        self.batch_size = batch_size
        self.dropout    = dropout
        self.workers    = workers

        if load_path is None: #Create a new model if no load path is provided.
            self.model = self.create_model()
        else:
            self.model = load_model(load_path)

    def train(self, train_sequence, validate_sequence):
        """Trains the network using the training set and validates it.

        Args:
            train_sequence (DataLoader): the training set data.
            validate_sequence (DataLoader): the validation set data.

        """
        learning_rate_reduction_cbk = ReduceLROnPlateau(
            monitor="val_loss",
            patience=10,
            verbose=1,
            factor=0.5,
            min_lr=0.000001,
        )

        self.history = self.model.fit(
            train_sequence,
            validation_data=validate_sequence,
            epochs=self.epochs,
            workers=self.workers,
            use_multiprocessing=False,
            verbose=1,
            callbacks=[learning_rate_reduction_cbk],
        )

    def test(self, test_sequence, test_labels, show_plots):
        """Evaluates the network against the test set and optionally, plots the confusion matrix.

        Args:
            test_sequence (DataLoader): the test set data to test against.
            test_labels (ndarray): an array of the labels for the test set.
            show_plots (Boolean): whether to show the confusion matrix plot.

        """
        print("Evaluating")
        loss, accuracy = self.model.evaluate(test_sequence)
        print("Test Loss: {0} | Test Accuracy: {1}".format(loss, accuracy))

        if show_plots:
            predictions = self.model.predict(test_sequence, use_multiprocessing=False, verbose=1)
            predictions = np.argmax(predictions, axis=1) #Get layer predictions
            remainder = len(test_labels) % self.batch_size

            # Sometimes batch_size may not divide evenly into number of samples
            # and so, some trimming may be required
            if remainder:
                test_labels = test_labels[:-remainder]

            labels = [i for i in "123"]
            cm = confusion_matrix(test_labels, predictions)
            df_cm = pd.DataFrame(cm, index=labels, columns=labels)
            ConfusionMatrixPrinter.pretty_plot(df_cm) #Plot the confusion matrix

    def save(self, save_path):
        """Saves the model under the given 'save_path'.

        Args:
            save_path (string): path to the directory to save the model to.

        """
        if not os.path.exists(save_path): #Make the required directory if not present.
            os.makedirs(save_path)
        self.model.save(os.path.join(save_path, "full_model.h5"))

    def create_model(self):
        """Creates the classifier model.

        Returns:
            A Keras Model object.

        """
        model = Sequential()
        model.add(Conv2D(32, (3,3), strides=(1,1), padding='same', activation="relu", input_shape=(*self.dims, self.channels)))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(64, (3,3), strides=(1,1), padding='same', activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(128, (3,3), strides=(1,1), padding='same', activation="relu"))
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
        model.add(Conv2D(32, (3,3), strides=(1,1), padding='same', activation="relu"))
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

        model.add(Flatten())
        model.add(Dense(300, activation="relu"))
        model.add(Dropout(self.dropout))
        model.add(Dense(192, activation="relu"))
        model.add(Dropout(self.dropout))
        model.add(Dense(123, activation="relu"))
        model.add(Dropout(self.dropout))
        model.add(Dense(79, activation="relu"))
        model.add(Dropout(self.dropout))
        model.add(Dense(50, activation="relu"))
        model.add(Dropout(self.dropout))
        model.add(Dense(3, activation="softmax")) #Number of layers as the number of softmax outputs.

        model.compile(
            optimizer=Nadam(self.lr),
            loss="sparse_categorical_crossentropy",
            metrics=["sparse_categorical_accuracy"]
        )
        return model

    def summary(self):
        """Displays a summary of the model."""
        self.model.summary()

def classify(data_path, save_path=None, load_path=None, train=True, summary=False, epochs=2,
         learning_rate=0.0003, batch_size=40, dropout_rate=0.1, workers=1, show_plots=True):
    """Either creates a classifier or loads an existing classifier, optionally
       trains the network and then evaluates it.

    Args:
        data_path (string): path to the directory containing the data to train and test on.
        save_path (string): path to the directory to save the trained model to.
        load_path (string): path to the full_model.h5 file to load an existing model from.
        train (Boolean): whether to train the model or not.
        summary (Boolean): whether to display a summary of the model or not.
        epochs (int): the number of epochs to train for.
        learning_rate (float): the value of the learning rate hyperparameter.
        batch_size (int): the size of each batch used when training.
        dropout_rate (float): the value of the dropout rate hyperparameter.
        workers (int): the number of workers to use.
        show_plots (Boolean): whether to display the classification confusion matrix after evaluation.

    """
    if save_path is not None: #If a save path is provided, save under the classifier directory
        save_path = os.path.join(save_path, "classifier")

    train_dir    = os.path.join(data_path, "train.h5")
    validate_dir = os.path.join(data_path, "validate.h5")
    test_dir     = os.path.join(data_path, "test.h5")

    train_file    = h5py.File(train_dir, "r")
    validate_file = h5py.File(validate_dir, "r")
    test_file     = h5py.File(test_dir, "r")

    labels = load_labels(data_path) #Get the labels for each split.
    #Subtract 1 from all labels to match with softmax output.
    train_labels, validate_labels, test_labels = labels["train"]-1, labels["validate"]-1, labels["test"]-1

    train_loader    = DataLoader(train_file,    train_labels,    DIMS, CHANNELS, batch_size, shuffle=False)
    validate_loader = DataLoader(validate_file, validate_labels, DIMS, CHANNELS, batch_size, shuffle=False)
    test_loader     = DataLoader(test_file,     test_labels,     DIMS, CHANNELS, batch_size, shuffle=False)

    model = Classifier(DIMS, CHANNELS, epochs, learning_rate, batch_size, dropout_rate, workers, load_path)
    if summary:
        model.summary()

    if train:
        model.train(train_loader, validate_loader)
    
    if save_path is not None:
        model.save(save_path)

    model.test(test_loader, test_labels, show_plots)

    train_file.close()
    validate_file.close()
    test_file.close()

def load_labels(path):
    """Loads the labels from train.h5, validate.h5 and test.h5 in the given directory.

    Args:
        path (string): path to the directory containing split data.

    Returns:
        Dictionary containing the labels for each split.

    """
    data = {}
    for section in ["train", "validate", "test"]:
        with h5py.File(os.path.join(path, "{}.h5".format(section)), "r") as f:
            data["{}".format(section)] = np.array(f["layers"]) #Get the number of layers for each file.
    return data


if __name__ == "__main__":
    data_path = "./models/neutron/data/merged"
    save_path = "./models/neutron"
    load_path = "./models/neutron/classifier/full_model.h5"

    #classify(data_path, save_path, train=True, epochs=20, show_plots=True) #Train new
    #classify(data_path, save_path, load_path=load_path, train=True, epochs=5, show_plots=True) #Train existing
    classify(data_path, load_path=load_path, train=False, show_plots=True) #Load existing but do not train
