import h5py, os
os.environ["KMP_AFFINITY"] = "none"

import numpy as np 
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Nadam

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

class Classifier():
    def __init__(self, dims, channels, epochs, lr, batch_size, dropout, workers, load_path=None):
        """ Initialisation"""
        self.dims       = dims
        self.channels   = channels
        self.epochs     = epochs
        self.lr         = lr
        self.batch_size = batch_size
        self.dropout    = dropout
        self.workers    = workers
        
        if load_path is None:
            self.model = self.create_model()
        else:
            self.model = load_model(load_path)

    def train(self, train_sequence, validate_sequence):
        """ Train and validate network"""
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
        return self.history

    def test(self, test_sequence, test_labels):
        print("Evaluating")
        loss, accuracy = self.model.evaluate(test_sequence)
        print("Test Loss: {0} | Test Accuracy: {1}".format(loss, accuracy))
        
        predictions = self.model.predict(test_sequence, use_multiprocessing=False, verbose=1)
        predictions = np.argmax(predictions, axis=1)
        remainder = len(test_labels) % self.batch_size

        # Sometimes batch_size may not divide evenly into number of samples
        # and so some trimming may be required
        if remainder:
            test_labels = test_labels[:-remainder]
            
        cm = confusion_matrix(test_labels, predictions)
        print("Confusion Matrix\n", cm)
        
    def save(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

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
        model.add(Dense(4, activation="softmax"))

        model.compile(
            optimizer=Nadam(self.lr),
            loss="sparse_categorical_crossentropy",
            metrics=["sparse_categorical_accuracy"]
        )
        return model
    
    def summary(self):
        self.model.summary()

def classify(data_path, save_path=None, load_path=None, train=True, summary=False, epochs=2, 
         learning_rate=0.0003, batch_size=40, dropout_rate=0.1, workers=1):
    
    if save_path is not None:
        save_path = os.path.join(save_path, "classifier")

    train_dir    = os.path.join(data_path, "train.h5")
    validate_dir = os.path.join(data_path, "validate.h5")
    test_dir     = os.path.join(data_path, "test.h5")

    train_file    = h5py.File(train_dir, "r")
    validate_file = h5py.File(validate_dir, "r")
    test_file     = h5py.File(test_dir, "r")

    labels = load_labels(data_path)
    train_labels, validate_labels, test_labels = labels["train"], labels["validate"], labels["test"]

    train_loader    = DataLoader(train_file,    train_labels,    DIMS, CHANNELS, batch_size, shuffle=False)
    validate_loader = DataLoader(validate_file, validate_labels, DIMS, CHANNELS, batch_size, shuffle=False)
    test_loader     = DataLoader(test_file,     test_labels,     DIMS, CHANNELS, batch_size, shuffle=False)
    
    model = Classifier(DIMS, CHANNELS, epochs, learning_rate, batch_size, dropout_rate, workers, load_path)
    if summary:
        model.summary()

    if train:
        model.train(train_loader, validate_loader)
        
    model.test(test_loader, test_labels)
    
    if save_path is not None:
        model.save(save_path)

    train_file.close()
    validate_file.close()
    test_file.close()

def load_labels(path):
    data = {}
    for section in ["train", "validate", "test"]:
        with h5py.File(os.path.join(path, "{}.h5".format(section)), "r") as f:
            data["{}".format(section)] = np.array(f["layers"])

    return data

if __name__ == "__main__":
    data_path = "./models/investigate/test/data/merged"
    save_path = "./models/investigate/test"
    #load_path = "./models/investigate/classification/classifier/full_model.h5"
    
    classify(data_path, save_path, load_path=None, train=True, epochs=2)
    