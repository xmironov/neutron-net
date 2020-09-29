import os, glob, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Suppress TensorFlow warnings

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.utils  import Sequence

from refnx.dataset  import ReflectDataset
from refnx.reflect  import SLD, ReflectModel
from refnx.analysis import Objective, CurveFitter

from generate_refnx import CurveGenerator
from generate_data  import generate_images, ImageGenerator, LAYERS_STR
from merge_data     import merge
from classification import classify, DIMS, CHANNELS
from regression     import regress

class DataLoaderClassification(Sequence):
    """DataLoaderClassification a Keras Sequence to load image data from a h5 file."""

    def __init__(self, labels, dim, channels, batch_size):
        """Initialises the DataLoaderClassification class with given parameters.

        Args:
            labels (ndarray): the labels corresponding to the loaded data.
            dim (tuple): dimensions of images loaded.
            channels (int): number of channels of images loaded.
            batch_size (int): size of each mini-batch to load.

        """
        self.labels     = labels
        self.dim        = dim
        self.channels   = channels
        self.batch_size = batch_size
        self.__on_epoch_end()

    def __len__(self):
        """Calculates the number of batches per epoch.

        Returns:
            An integer number of batches per epoch.

        """
        return int(np.floor(len(self.labels.keys()) / self.batch_size))

    def __getitem__(self, index):
        """Generates one batch of data.

        Args:
            index (int): position of batch.

        Returns:
            A `batch_size` sample of images (inputs) and classes (targets).

        """
        indices = self.indices[index * self.batch_size: (index + 1) * self.batch_size]
        indices = [list(self.labels.keys())[k] for k in indices]
        images, targets = self.__data_generation(indices)
        return images, targets

    def __data_generation(self, indices):
        """Generates data containing batch_size samples.

        Args:
            indices (ndarray): an array of indices to retrieve data from.

        Returns:
            A `batch_size` sample of images (inputs) and classes (targets).

        """
        images  = np.empty((self.batch_size, *self.dim, self.channels))
        classes = np.empty((self.batch_size, 1), dtype=int)

        for i, np_image_filename in enumerate(indices): #Get images and classes for each index
            images[i,]  = np.load(np_image_filename)
            classes[i,] = self.labels[np_image_filename]

        return images, classes

    def __on_epoch_end(self):
        """Updates indices after each epoch."""
        self.indices = np.arange(len(self.labels.keys()))


class DataLoaderRegression(Sequence):
    """DataLoaderRegression uses a Keras Sequence to load image data from a h5 file."""

    def __init__(self, labels_dict, dim, channels):
        """Initialises the DataLoaderRegression class with given parameters.

        Args:
            labels_dict (dict): dictionary containing labels for each file.
            dim (tuple): dimensions of images loaded.
            channels (int): number of channels of images loaded.

        """
        self.labels_dict = labels_dict
        self.dim         = dim
        self.channels    = channels
        self.__on_epoch_end()

    def __len__(self):
        """Calculates the number of batches per epoch.

        Returns:
            An integer number of batches per epoch.

        """
        return int(np.floor(len(self.labels_dict.keys()))) #batch_size is set as 1

    def __getitem__(self, index):
        """Generates one batch of data.

        Args:
            index (int): position of batch.

        Returns:
            A sample of images (inputs).

        """
        indices = self.indices[index: (index + 1)]
        indices = [list(self.labels_dict.keys())[k] for k in indices]
        images  = self.__data_generation(indices)
        return images

    def __data_generation(self, indices):
        """Generates data containing batch_size samples.

        Args:
            indices (ndarray): an array of indices to retrieve data from.

        Returns:
            A `batch_size` sample of images (inputs) and targets.

        """
        images = np.empty((1, *self.dim, self.channels))
        layers = np.empty((1, 1))

        for i, np_image_filename in enumerate(indices):
            images[i,] = np.load(np_image_filename)
            layers[i,] = self.labels_dict[np_image_filename]["class"]

        return images, layers

    def __on_epoch_end(self):
        """Updates indices after each epoch."""
        self.indices = np.arange(len(self.labels_dict.keys()))


class KerasDropoutPredicter():
    """KerasDropoutPredicter takes trained models and uses dropout at test time to make Bayesian-like predictions."""

    def __init__(self, models):
        """Initialises the dropout predictor with given models for each layer.

        Args:
            models (list): a list of Keras models.

        """
        # One-layer model function
        self.f_1 = K.function([models[1].layers[0].input, K.learning_phase()],
                              [models[1].layers[-2].output, models[1].layers[-1].output])

        # Two-layer model function
        self.f_2 = K.function([models[2].layers[0].input, K.learning_phase()],
                              [models[2].layers[-2].output, models[2].layers[-1].output])

        if len(models) == 3:
            # Three-layer model function
            self.f_3 = K.function([models[3].layers[0].input, K.learning_phase()],
                                  [models[3].layers[-2].output, models[3].layers[-1].output])

    def predict(self, sequence, n_iter=5):
        """Makes Bayesian-like predictions using given models.

        Args:
            sequence (DataLoaderRegression): the sequence providing data to predict on.
            n_iter (int): the number of iterations per step.

        Returns:
            List of depth and SLD predictions along with errors associated with each.

        """
        steps_done = 0
        all_out = []
        steps = len(sequence)
        output_generator = KerasDropoutPredicter.__iter_sequence_infinite(sequence)

        while steps_done < steps:
            # Yield the sample image and the number of layers it is predicted to have
            x, y = next(output_generator)
            results = []

            for i in range(n_iter):
                if y[0][0] == 1: # If one-layer
                    [depths, slds] = self.f_1([x, 1])
                elif y[0][0] == 2: #Else if two-layer
                    [depths, slds] = self.f_2([x, 1])
                elif y[0][0] == 3: #Else if three-layer
                    [depths, slds] = self.f_3([x, 1])
                    
                depth_scaled = ImageGenerator.scale_to_range(depths, (0, 1), ImageGenerator.depth_bounds)
                sld_scaled   = ImageGenerator.scale_to_range(slds,   (0, 1), ImageGenerator.sld_bounds)
                results.append([depth_scaled, sld_scaled])

            results = np.array(results)
            prediction, uncertainty = results.mean(axis=0), results.std(axis=0)
            outs = np.array([prediction, uncertainty])
            if not all_out:
                for out in outs:
                    all_out.append([])

            for i, out in enumerate(outs):
                all_out[i].append(out)

            steps_done += 1

        return [np.concatenate(out, axis=1) for out in all_out]

    @staticmethod
    def __iter_sequence_infinite(sequence):
        """Provides an infinite iterator for given sequence.

        Args:
            sequence (DataLoaderRegression): the sequence providing data to predict on.

        """
        while True:
            for item in sequence:
                yield item


class Model():
    """The Model class represents a refnx model using predictions made by the classifier and regressors.

    Class Attributes:
        si_sld (float): the substrate SLD (silicon).
        roughness (int): the default roughness between each layer in Angstrom.
        dq (int): the instrument resolution parameter.
        scale (int): the instrument scale parameter.

    """
    si_sld    = 2.047
    roughness = 2
    dq        = 2
    scale     = 1

    def __init__(self, file_path, layers, predicted_slds, predicted_depths):
        """Initalises the Model class by creating a refnx model with given predicted values.

        Args:
            file_path (string): a path to the file with the data to construct the model for.
            layers (int): the number of layers for the model predicted by the classifier.
            predicted_slds (ndarray): an array of predicted SLDs for each layer.
            predicted_depths (ndarray): an array of predicted depths for each layer.

        """
        self.structure = SLD(0, name='Air') #Model starts with air.

        for i in range(layers):
            layer = SLD(predicted_slds[i], name='Layer {}'.format(i+1))(thick=predicted_depths[i], rough=Model.roughness)
            layer.sld.real.setp(bounds=ImageGenerator.sld_bounds, vary=True)
            layer.thick.setp(bounds=ImageGenerator.depth_bounds,  vary=True)
            self.structure = self.structure | layer  #Next comes each layer.

        si_substrate = SLD(Model.si_sld, name='Si Substrate')(thick=0, rough=Model.roughness) #Then substrate
        self.structure = self.structure | si_substrate

        data = ReflectDataset(file_path) #Load the data for which the model is designed for.
        self.model = ReflectModel(self.structure, scale=Model.scale, dq=Model.dq)
        self.objective = Objective(self.model, data)

    def fit(self):
        """Fits the model to the data using differential evolution"""
        fitter = CurveFitter(self.objective)
        fitter.fit('differential_evolution', verbose=False)
        self.plot_objective(prediction=False)

    def plot_SLD(self):
        """Plots the SLD profile for the model."""
        plt.figure()
        plt.plot(*self.structure.sld_profile())
        plt.ylabel('SLD /$10^{-6} \AA^{-2}$')
        plt.xlabel('distance / $\AA$')

    def plot_reflectivity(self, qMin=0.005, qMax=0.3, points=1000):
        """Plots the reflectivity profile for the model.

        Args:
            qMin (int): the minimum q value to use when generating r values.
            qMax (int): the maximum q value to use when generating r values.
            points (int): the number of q values to use.

        """
        q = np.linspace(qMin, qMax, points)
        plt.figure()
        plt.plot(q, self.model(q))
        plt.xlabel('Q')
        plt.ylabel('Reflectivity')
        plt.yscale('log')

    def plot_objective(self, prediction=True):
        """Plots the current objective for the model against given dataset.

        Args:
            prediction (Boolean): whether the plot is a prediction or fit.

        """
        if prediction:
            title='Reflectivity Plot using Predicted Values'
            label="Predicted"
        else:
            title='Reflectivity Plot using Fitted Values'
            label="Fitted"
        fig = plt.figure(figsize=[9,7], dpi=200)
        ax = fig.add_subplot(111)
        
        y, y_err, model = self.objective._data_transform(model=self.objective.generative())
        # Add the data in a transformed fashion.
        ax.errorbar(self.objective.data.x, y, y_err, label=self.objective.data.name,
                    color="blue", marker="o", ms=3, lw=0, elinewidth=1, capsize=1.5)
        #Add the fit
        ax.plot(self.objective.data.x, model, color="red", label=label, zorder=20)
        
        plt.xlabel('Q', fontsize=11, weight='bold')
        plt.ylabel('Reflectivity', fontsize=11, weight='bold')
        plt.yscale('log')
        plt.legend()
        if title:
            plt.title(title, fontsize=15, pad=15)


class Pipeline:
    """The Pipeline class can perform data generation, training and predictions."""

    @staticmethod
    def run(data_path, save_path, classifier_path, regressor_paths, fit=True, n_iter=5):
        """Performs classification and regression to create a refnx model for given .dat files.

        Args:
            data_path (string): path to the directory containing .dat files for predicting on.
            save_path (string): path to the directory where temporary files are to be stored.
            classifier_path (string): path to a pre-trained classifier.
            regressor_paths (dict): dictionary of paths to regressors for each layer.
            fit (Boolean): whether to fit the newly generated models.
            n_iter (int): number of times to predict using the KDP.

        Returns:
            A dictionary of models, index by filename, initialized with predicted values for each .dat file.

        """
        dat_files = glob.glob(os.path.join(data_path, '*.dat')) #Search for .dat files.

        #Classify the number of layers for each .dat file.
        layer_predictions, npy_image_filenames = Pipeline.__classify(dat_files, save_path, classifier_path)
        for curve in range(len(dat_files)):
            filename = os.path.basename(dat_files[curve])
            print("Results for '{}'".format(filename))
            print(">>> Predicted number of layers: {}\n".format(layer_predictions[curve]))

        #Use regression to predict the SLDs and depths for each file.
        sld_predictions, depth_predictions, sld_errors, depth_errors = Pipeline.__regress(data_path, save_path, regressor_paths, layer_predictions, npy_image_filenames, n_iter)
        
        models = {} 
        #Print the predictions and errors for the depths and SLDs for each layer for each file.
        for curve in range(len(dat_files)): #Iterate over each file.
            filename = os.path.basename(dat_files[curve])
            print("Results for '{}'".format(filename))
            for i in range(layer_predictions[curve]): #Iterate over each layer
                print(">>> Predicted layer {0} - SLD:   {1:10.4f} | Error: {2:10.6f}".format(i+1, sld_predictions[curve][i], sld_errors[curve][i]))
                print(">>> Predicted layer {0} - Depth: {1:10.4f} | Error: {2:10.6f}".format(i+1, depth_predictions[curve][i], depth_errors[curve][i]))

            #Create a refnx model with the predicted number of layers, SLDs and depths.
            model = Model(dat_files[curve], layer_predictions[curve], sld_predictions[curve], depth_predictions[curve])
            model.plot_objective(prediction=True)
            models[filename] = model
            print()
            
        if fit: #Fit each model if requested.
            Pipeline.__fit(models)
            
        return models

    @staticmethod
    def __classify(dat_files, save_path, classifier_path):
        """Performs layer classification for specified .dat files.

        Args:
            dat_files (list): a list of file paths for .dat files to predict on.
            save_path (type): path to the directory where temporary files are to be stored.
            classifier_path (type): path to a pre-trained classifier.

        Returns:
            The layer predictions for each file along with Numpy image filenames.

        """
        print("-------------- Classification -------------")
        #Convert .dat files to images, ready for passing as input to the classifier.
        npy_image_filenames = Pipeline.__dat_files_to_npy_images(dat_files, save_path)
        class_labels = dict(zip(npy_image_filenames, np.zeros((len(npy_image_filenames), 1))))

        classifier_loader = DataLoaderClassification(class_labels, DIMS, CHANNELS, 1)
        classifier = load_model(classifier_path)
        return np.argmax(classifier.predict(classifier_loader, verbose=1), axis=1), npy_image_filenames #Make predictions

    @staticmethod
    def __regress(data_path, save_path, regressor_paths, layer_predictions, npy_image_filenames, n_iter):
        """Performs SLD and depth regression for specified .dat files.

        Args:
            data_path (string): path to the directory containing .dat files for predicting on.
            save_path (string): path to the directory where temporary files are to be stored.
            regressor_paths (dict): dictionary of paths to regressors for each layer.
            layer_predictions (ndarray): an array of layer predictions for each file.
            npy_image_filenames (ndarray): an array of filenames of files containing images
                                           corresponding to the input .dat files.
            n_iter (int): number of times to predict using the KDP.
                              
        Returns:
            SLD and depth predictions along with the errors for each.

        """
        print("---------------- Regression ---------------")
        # Dictionary to pair image with "depth", "sld", "class" values
        values_labels = {}
        for filename, layer_prediction in zip(npy_image_filenames, layer_predictions):
            values_labels[filename] = {"depth": np.zeros((1, int(layer_prediction))),
                                       "sld":   np.zeros((1, int(layer_prediction))),
                                       "class":              int(layer_prediction)}

        loader = DataLoaderRegression(values_labels, DIMS, CHANNELS)
        regressors = {layer: load_model(regressor_paths[layer]) for layer in regressor_paths.keys()}

        #Use custom class to activate Dropout at test time in models
        kdp = KerasDropoutPredicter(regressors)
        kdp_predictions = kdp.predict(loader, n_iter=n_iter)

        #Predictions given as [depth_1, depth_2, depth_3], [sld_1, sld_2, sld_3]
        depth_predictions = kdp_predictions[0][0]
        sld_predictions   = kdp_predictions[0][1]

        #Errors given as [depth_std_1, depth_std_2], [sld_std_1, sld_std_2]
        depth_errors = kdp_predictions[1][0]
        sld_errors   = kdp_predictions[1][1]

        return sld_predictions, depth_predictions, sld_errors, depth_errors

    @staticmethod
    def __fit(models):
        """Performs fitting on the given models.

        Args:
            models (dict): a dictionary of refnx models, index by filename.

        """
        print("----------------- Fitting -----------------")
        for filename in models.keys(): #Iterate over each model and fit.
            print("Results for '{}'".format(filename))
            model = models[filename]
            model.fit()
            for i, component in enumerate(model.structure.components[1:-1]): #Iterate over each layer
                print(">>> Fitted layer {0} - SLD:   {1:10.4f}".format(i+1, component.sld.real.value))
                print(">>> Fitted layer {0} - Depth: {1:10.4f}".format(i+1, component.thick.value))
            print()

    @staticmethod
    def __dat_files_to_npy_images(dat_files, save_path):
        """Given a list of .dat files, creates .npy images and save them in `save_path`.

        Args:
            dat_files (list): a list of .dat file paths.
            save_path (string): the path to the directory to store npy images in.

        Returns:
            An array of filenames of files containing images corresponding to the input .dat files.

        """
        if dat_files == []:
            sys.exit("No .dat files found for classification in save path")

        image_files = []
        for dat_file in dat_files:
            # Identify if there are column headings, or whether the header is empty
            header_setting = Pipeline.__identify_header(dat_file)

            if header_setting is None:
                data = pd.read_csv(dat_file, header=0, delim_whitespace=True, names=['X', 'Y', 'Error'])
            else:
                data = pd.read_csv(dat_file, header=header_setting)

            head, tail = os.path.split(dat_file)
            name = os.path.normpath(os.path.join(save_path, tail)).replace(".dat", ".npy")
            image_files.append(name)
            sample_momentum = data["X"]
            sample_reflect  = data["Y"]
            sample = np.vstack((sample_momentum, sample_reflect)).T
            img = ImageGenerator.image_process(sample) #Convert the reflectivity data to an image.
            np.save(name, img)

        return image_files

    @staticmethod
    def __identify_header(path, threshold=0.9):
        """Parses the .dat file header to find out if there are headings or if it's empty.

        Args:
            path (string): file path to a .dat file
            threshold (float): a threshold constant for the similarity check.

        Returns:
            None or 'infer' based on whether a header is present or not respectively.

        """
        dataframe1 = pd.read_csv(path, header='infer', nrows=5)
        dataframe2 = pd.read_csv(path, header=None,    nrows=5)
        similarity = (dataframe1.dtypes.values == dataframe2.dtypes.values).mean()
        return 'infer' if similarity < threshold else None

    @staticmethod
    def setup(save_path, layers=[1,2,3], curve_num=5000, chunk_size=1000, show_plots=True, generate_data=True,
              train_classifier=True, train_regressor=True, classifer_epochs=2, regressor_epochs=2):
        """Sets up the pipeline for predictions on .dat files by generating data and training.

        Args:
            save_path (string): a path to the directory where data and models will be saved.
            layers (list): a list of layers to generate and train for.
            curve_num (int): the number of curves to generate per layer.
            chunk_size (int): the size of chunks to use in the h5 storage of images for curves.
            show_plots (Boolean): whether to display classification confusion matrix and regression plots or not.
            generate_data (Boolean): whether to generate data or use existing data.
            train_classifier (Boolean): whether to train the classifier or not.
            train_regressor (Boolean): whether to train the regressors or not.
            classifer_epochs (int): the number of epochs to train the classifier for.
            regressor_epochs (int): the number of epochs to train the regressor for.

        """
        if generate_data:
            print("-------------- Data Generation ------------")
            for layer in layers: #Generate curves for each layer specified.
                print(">>> Generating {}-layer curves".format(layer))
                structures = CurveGenerator.generate(curve_num, layer, sld_bounds=(-0.5,10), thick_bounds=(20,3000), substrate_SLD=2.047)
                CurveGenerator.save(save_path + "/data", LAYERS_STR[layer], structures) #Save the generated curves.

                print(">>> Creating images for {}-layer curves".format(layer))
                save_path_layer = data_path_layer = save_path + "/data/{}".format(LAYERS_STR[layer])
                #Create images for the generated curves, ready for input to the classifier and regressors.
                generate_images(data_path_layer, save_path_layer, [layer], chunk_size=chunk_size, display_status=False)

            layers_paths = [save_path + "/data/{}".format(LAYERS_STR[layer]) for layer in layers]
            merge(save_path + "/data", layers_paths) #Merge the curves for each layer for classification.

        print("\n-------------- Classification -------------")
        if train_classifier:
            print(">>> Training classifier")
            classify(save_path + "/data/merged", save_path, train=True, epochs=classifer_epochs, show_plots=show_plots) #Train the classifier.
        else:
            print(">>> Loading classifier")
            load_path = save_path + "/classifier/full_model.h5" #Load a classifier.
            classify(save_path + "/data/merged", load_path=load_path, train=False, show_plots=show_plots)

        print("\n---------------- Regression ---------------")
        for layer in layers: #Train or load regressors for each layer that we are setting up for.
            data_path_layer = save_path + "/data/{}".format(LAYERS_STR[layer])
            if train_regressor:
                print(">>> Training {}-layer regressor".format(LAYERS_STR[layer]))
                regress(data_path_layer, layer, save_path, epochs=regressor_epochs, show_plots=show_plots) #Train the regressor.
            else:
                print(">>> Loading {}-layer regressor".format(LAYERS_STR[layer]))
                load_path_layer = save_path + "/{}-layer-regressor/full_model.h5".format(LAYERS_STR[layer]) #Load an existing regressor.
                regress(data_path_layer, layer, load_path=load_path_layer, train=False, show_plots=show_plots)
            print()

if __name__ == "__main__":
    save_path = './models/deploy'
    layers     = [1, 2, 3]
    curve_num  = 25000
    chunk_size = 1000
    show_plots       = True
    generate_data    = True
    train_classifier = True
    train_regressor  = True
    #Pipeline.setup(save_path, layers, curve_num, chunk_size, show_plots, generate_data, 
    #               train_classifier, train_regressor, classifer_epochs=20, regressor_epochs=15)

    load_path = "./models/deploy"
    data_path = "./models/deploy"
    classifier_path = load_path + "/classifier/full_model.h5"
    layers = 2
    regressor_paths = {i: load_path + "/{}-layer-regressor/full_model.h5".format(LAYERS_STR[i]) for i in range(1, layers+1)}
    models = Pipeline.run(data_path, save_path, classifier_path, regressor_paths, fit=True, n_iter=100)
