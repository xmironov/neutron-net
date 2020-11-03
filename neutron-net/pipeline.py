import os, glob, sys
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.utils  import Sequence
tf.compat.v1.disable_eager_execution()

from refnx.dataset  import ReflectDataset
from refnx.reflect  import SLD, MaterialSLD, ReflectModel
from refnx.analysis import Objective, CurveFitter

from generate_refnx import CurveGenerator, NeutronGenerator, XRayGenerator
from generate_data  import generate_images, ImageGenerator, LAYERS_STR, DIMS, CHANNELS
from merge_data     import merge
from classification import classify
from regression     import regress

class DataLoaderClassification(Sequence):
    """DataLoaderClassification a Keras Sequence to load image data from a h5 file."""

    def __init__(self, labels, dim, channels):
        """Initialises the DataLoaderClassification class with given parameters.

        Args:
            labels (ndarray): the labels corresponding to the loaded data.
            dim (tuple): dimensions of images loaded.
            channels (int): number of channels of images loaded.

        """
        self.labels     = labels
        self.dim        = dim
        self.channels   = channels
        self.__on_epoch_end()

    def __len__(self):
        """Calculates the number of batches.

        Returns:
            An integer number of batches.

        """
        return len(self.labels.keys())

    def __getitem__(self, index):
        """Generates one batch of data.

        Args:
            index (int): position of batch.

        Returns:
            A single sample of images (inputs) and classes (targets).

        """
        indices = self.indices[index: (index + 1)]
        indices = [list(self.labels.keys())[k] for k in indices]
        images, targets = self.__data_generation(indices)
        return images, targets

    def __data_generation(self, indices):
        """Generates data containing 1 sample.

        Args:
            indices (ndarray): an array of indices to retrieve data from.

        Returns:
            An image (input) and its class (target).

        """
        images  = np.empty((1, *self.dim, self.channels))
        classes = np.empty((1, 1), dtype=int)

        for i, np_image_filename in enumerate(indices): #Get images and classes for each index
            images[i,]  = np.load(np_image_filename)
            classes[i,] = self.labels[np_image_filename]

        return images, classes

    def __on_epoch_end(self):
        """Updates indices after each epoch."""
        self.indices = np.arange(len(self.labels.keys()))


class DataLoaderRegression(Sequence):
    """DataLoaderRegression uses a Keras Sequence to load image data from a Numpy image file."""

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
        """Calculates the number of batches.

        Returns:
            An integer number of batches.

        """
        return len(self.labels_dict.keys()) #batch_size is set as 1

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
        """Generates data containing 1 sample.

        Args:
            indices (ndarray): an array of indices to retrieve data from.

        Returns:
            A sample of images (inputs) and its corresponding layer.

        """
        images = np.empty((1, *self.dim, self.channels))
        layers = np.empty((1, 1))

        for i, np_image_filename in enumerate(indices):
            images[i,] = np.load(np_image_filename)
            layers[i,] = self.labels_dict[np_image_filename]["class"]

        return images, layers

    def __on_epoch_end(self):
        """Updates indices."""
        self.indices = np.arange(len(self.labels_dict.keys()))


class KerasDropoutPredicter():
    """KerasDropoutPredicter takes trained models and uses dropout at test time to make Bayesian-like predictions."""

    def __init__(self, models):
        """Initialises the dropout predictor with given models for each layer.

        Args:
            models (list): a list of Keras models.

        """
        if 1 in models: #One-layer model function
            self.f_1 = K.function([models[1].layers[0].input, K.learning_phase()],
                                  [models[1].layers[-2].output, models[1].layers[-1].output])

        if 2 in models: #Two-layer model function
            self.f_2 = K.function([models[2].layers[0].input, K.learning_phase()],
                                  [models[2].layers[-2].output, models[2].layers[-1].output])

        if 3 in models: #Three-layer model function
            self.f_3 = K.function([models[3].layers[0].input, K.learning_phase()],
                                  [models[3].layers[-2].output, models[3].layers[-1].output])

    def predict(self, sequence, n_iter=5, xray=False):
        """Makes Bayesian-like predictions using given models.

        Args:
            sequence (DataLoaderRegression): the sequence providing data to predict on.
            n_iter (int): the number of iterations per step.
            xray (Boolean): whether data to predict on uses a neutron or x-ray probe.

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
            layer = int(y[0][0])
            results = []

            for i in range(n_iter): #Run the corresponding model for the predicted layer.
                if layer == 1:
                    [depths, slds] = self.f_1([x, 1])
                elif layer == 2:
                    [depths, slds] = self.f_2([x, 1])
                elif layer == 3:
                    [depths, slds] = self.f_3([x, 1])

                depth_scaled = np.zeros((1,3)) #Pad empty depth and SLD values for layers not being predicted on.
                sld_scaled   = np.zeros((1,3))

                depth_scaled[0][:layer] = ImageGenerator.scale_to_range(depths, (0, 1), ImageGenerator.depth_bounds)
                if xray:
                    sld_scaled[0][:layer]  = ImageGenerator.scale_to_range(slds, (0, 1), ImageGenerator.sld_xray_bounds)
                else:
                    sld_scaled[0][:layer]  = ImageGenerator.scale_to_range(slds, (0, 1), ImageGenerator.sld_neutron_bounds)

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
        roughness (int): the default roughness between each layer in Angstrom.
        rough_bounds (tuple): the range of values to fit for roughness.
        si_sld (float): the substrate SLD (silicon).
        dq (float): the instrument resolution parameter.
        scale (float): the instrument scale parameter.
        bkg (float): value for the background parameter.

    """
    roughness    = 8
    rough_bounds = CurveGenerator.rough_bounds
    si_sld       = NeutronGenerator.substrate_sld
    dq           = CurveGenerator.dq
    scale        = CurveGenerator.scale
    bkg          = 2e-7

    def __init__(self, file_path, layers, predicted_slds, predicted_depths, xray):
        """Initialises the Model class by creating a refnx model with given predicted values.

        Args:
            file_path (string): a path to the file with the data to construct the model for.
            layers (int): the number of layers for the model predicted by the classifier.
            predicted_slds (ndarray): an array of predicted SLDs for each layer.
            predicted_depths (ndarray): an array of predicted depths for each layer.
            xray (Boolean): whether the model should use a neutron or x-ray probe.

        """
        self.structure = SLD(0, name='Air') #Model starts with air.

        if xray: #Use x-ray probe
            for i in range(layers):
                density = predicted_slds[i] / XRayGenerator.density_constant
                SLD_layer = MaterialSLD(XRayGenerator.material, density, probe='x-ray', wavelength=XRayGenerator.wavelength, name='Layer {}'.format(i+1))
                layer = SLD_layer(thick=predicted_depths[i], rough=Model.roughness)
                layer.density.setp(bounds=XRayGenerator.density_bounds, vary=True)
                layer.thick.setp(bounds=ImageGenerator.depth_bounds, vary=True)
                layer.rough.setp(bounds=Model.rough_bounds, vary=True)
                self.structure = self.structure | layer  #Next comes each layer.
            #Then substrate
            si_substrate = MaterialSLD(XRayGenerator.material, XRayGenerator.substrate_density, probe='x-ray', name='Si Substrate')(thick=0, rough=Model.roughness)

        else: #Use neutron probe
            for i in range(layers):
                layer = SLD(predicted_slds[i], name='Layer {}'.format(i+1))(thick=predicted_depths[i], rough=Model.roughness)
                layer.sld.real.setp(bounds=ImageGenerator.sld_neutron_bounds, vary=True)
                layer.thick.setp(bounds=ImageGenerator.depth_bounds, vary=True)
                layer.rough.setp(bounds=Model.rough_bounds, vary=True)
                self.structure = self.structure | layer  #Next comes each layer.
            #Then substrate
            si_substrate = SLD(Model.si_sld, name='Si Substrate')(thick=0, rough=Model.roughness)

        si_substrate.rough.setp(bounds=Model.rough_bounds, vary=True)
        self.structure = self.structure | si_substrate
        
        data = self.__load_data(file_path) #Pre-process and load given dataset.
        self.model = ReflectModel(self.structure, scale=Model.scale, dq=Model.dq, bkg=Model.bkg)
        self.objective = Objective(self.model, data)

    def __load_data(self, file_path):
        """Loads a reflectivity dataset from a given file path and applies scaling.

        Args:
            file_path (string): a path to the file with the data to construct the model for.

        """
        data = ReflectDataset(file_path) #Load the data for which the model is designed for.
        self.filename = os.path.basename(data.filename)
        data.scale(np.max(data.data[1])) #Normalise Y and Error by dividing by max R point.
        
        x, y, y_err = data.x.tolist(), data.y.tolist(), data.y_err.tolist()
        removed = [] #Remove any points containing 0 values as these cause NaNs when fitting.
        for i in range(len(x)):
            if x[i] == 0 or y[i] == 0 or y_err[i] == 0:
                removed.append(i)
        
        #Remove the identified points and return the processed dataset.
        x     = np.delete(np.array(x),     removed)
        y     = np.delete(np.array(y),     removed)
        y_err = np.delete(np.array(y_err), removed)
        data_new = np.array([x, y, y_err])
        return ReflectDataset(data_new)

    def fit(self):
        """Fits the model to the data using differential evolution."""
        fitter = CurveFitter(self.objective)
        fitter.fit('differential_evolution', verbose=False)
        fitter.sample(400)
        fitter.reset()
        fitter.sample(15, nthin=100)

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
        fig = plt.figure(figsize=[9,7], dpi=600)
        ax = fig.add_subplot(111)

        y, y_err, model = self.objective._data_transform(model=self.objective.generative())
        # Add the data in a transformed fashion.
        ax.errorbar(self.objective.data.x, y, y_err, label=self.filename,
                    color="blue", marker="o", ms=3, lw=0, elinewidth=1, capsize=1.5)
        #Add the prediction/fit
        ax.plot(self.objective.data.x, model, color="red", label=label, zorder=20)

        plt.xlabel("$\mathregular{Q\ (Å^{-1})}$", fontsize=11, weight='bold')
        plt.ylabel('Reflectivity (arb.)',         fontsize=11, weight='bold')
        plt.yscale('log')
        plt.legend()
        if title:
            plt.title(title, fontsize=15, pad=15)


class Pipeline:
    """The Pipeline class can perform data generation, training and predictions."""

    @staticmethod
    def run(data_path, save_path, classifier_path, regressor_paths, fit=True, n_iter=100, xray=False):
        """Performs classification and regression to create a refnx model for given .dat files.

        Args:
            data_path (string): path to the directory containing .dat files for predicting on.
            save_path (string): path to the directory where temporary files are to be stored.
            classifier_path (string): path to a pre-trained classifier.
            regressor_paths (dict): dictionary of paths to regressors for each layer.
            fit (Boolean): whether to fit the newly generated models.
            n_iter (int): number of times to predict using the KDP.
            xray (Boolean): whether the .dat files are x-ray or neutron.

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
        sld_predictions, depth_predictions, sld_errors, depth_errors = Pipeline.__regress(data_path, regressor_paths, layer_predictions, npy_image_filenames, n_iter, xray)

        models = {}
        #Print the predictions and errors for the depths and SLDs for each layer for each file.
        for curve in range(len(dat_files)): #Iterate over each file.
            filename = os.path.basename(dat_files[curve])
            print("Results for '{}'".format(filename))
            for i in range(layer_predictions[curve]): #Iterate over each layer
                print(">>> Predicted layer {0} - SLD:   {1:9.3f} | Error: {2:7.3f}".format(i+1, sld_predictions[curve][i], sld_errors[curve][i]))
                print(">>> Predicted layer {0} - Depth: {1:9.4f} | Error: {2:7.3f}".format(i+1, depth_predictions[curve][i], depth_errors[curve][i]))

            #Create a refnx model with the predicted number of layers, SLDs and depths.
            model = Model(dat_files[curve], layer_predictions[curve], sld_predictions[curve], depth_predictions[curve], xray)
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
        print("---------------------- Classification --------------------")
        #Convert .dat files to images, ready for passing as input to the classifier.
        npy_image_filenames = Pipeline.dat_files_to_npy_images(dat_files, save_path)
        class_labels = dict(zip(npy_image_filenames, np.zeros((len(npy_image_filenames), 1))))

        classifier_loader = DataLoaderClassification(class_labels, DIMS, CHANNELS)
        classifier = load_model(classifier_path)
        return [1,1], npy_image_filenames
        #return np.argmax(classifier.predict(classifier_loader, verbose=1), axis=1)+1, npy_image_filenames #Make predictions

    @staticmethod
    def __regress(data_path, regressor_paths, layer_predictions, npy_image_filenames, n_iter, xray=False):
        """Performs SLD and depth regression for specified .dat files.

        Args:
            data_path (string): path to the directory containing .dat files for predicting on.
            regressor_paths (dict): dictionary of paths to regressors for each layer.
            layer_predictions (ndarray): an array of layer predictions for each file.
            npy_image_filenames (ndarray): an array of filenames of files containing images
                                           corresponding to the input .dat files.
            n_iter (int): number of times to predict using the KDP.
            xray (Boolean): whether the .dat files are x-ray or neutron.

        Returns:
            SLD and depth predictions along with the errors for each.

        """
        print("----------------------- Regression -----------------------")
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
        kdp_predictions = kdp.predict(loader, n_iter, xray)

        depth_predictions = kdp_predictions[0][0]
        sld_predictions   = kdp_predictions[0][1]

        depth_errors = kdp_predictions[1][0]
        sld_errors   = kdp_predictions[1][1]

        return sld_predictions, depth_predictions, sld_errors, depth_errors

    @staticmethod
    def __fit(models):
        """Performs fitting on the given models.

        Args:
            models (dict): a dictionary of refnx models, index by filename.

        """
        print("------------------------- Fitting ------------------------")
        for filename in models.keys(): #Iterate over each model and fit.
            print("Results for '{}'".format(filename))
            model = models[filename]
            model.fit()
            model.plot_objective(prediction=False)

            for i, component in enumerate(model.structure.components[1:-1]): #Iterate over each layer
                print(">>> Fitted layer {0} - SLD:   {1:9.3f} | Error: {2:7.5f}".format(i+1, component.sld.real.value, component.sld.real.stderr))
                print(">>> Fitted layer {0} - Depth: {1:9.3f} | Error: {2:7.5f}".format(i+1, component.thick.value,    component.thick.stderr))
            print()

    @staticmethod
    def dat_files_to_npy_images(dat_files, save_path):
        """Given a list of .dat files, creates .npy images and save them in `save_path`.

        Args:
            dat_files (list): a list of .dat file paths.
            save_path (string): the path to the directory to store npy images in.

        Returns:
            An array of filenames of files containing images corresponding to the input .dat files.

        """
        if dat_files == []:
            sys.exit("No .dat files found for classification in save path")
        temp_path = save_path + '/npy_images'
        image_files = []
        for dat_file in dat_files:
            data = pd.read_csv(dat_file, header='infer', sep='\s+', names=['X', 'Y', 'Error'])
            data = data[(data != 0).all(1)] #Remove any 0 values.
            
            if not os.path.exists(temp_path):
                os.makedirs(temp_path)
            head, tail = os.path.split(dat_file)
            name = os.path.normpath(os.path.join(temp_path, tail)).replace(".dat", ".npy")
            image_files.append(name)
            sample_momentum = data["X"]
            sample_reflect  = data["Y"]
    
            sample_reflect_norm = sample_reflect / np.max(sample_reflect) #Normalise data so that max reflectivity is 1

            sample = np.vstack((sample_momentum, sample_reflect_norm)).T
            img = ImageGenerator.image_process(sample, save_format=False) #Convert the reflectivity data to an image.
            np.save(name, img)

        return image_files

    @staticmethod
    def setup(save_path, layers=[1,2,3], curve_num=5000, chunk_size=1000, noisy=False, xray=False,
              show_plots=True, generate_data=True, train_classifier=True, train_regressor=True,
              classifer_epochs=2, regressor_epochs=2):
        """Sets up the pipeline for predictions on .dat files by generating data and training.

        Args:
            save_path (string): a path to the directory where data and models will be saved.
            layers (list): a list of layers to generate and train for.
            curve_num (int): the number of curves to generate per layer.
            chunk_size (int): the size of chunks to use in the h5 storage of images for curves.
            noisy (Boolean): whether to add noise to generated data.
            xray (Boolean): whether to use an x-ray probe or not.
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
                if xray: #Generate data using x-ray probe.
                    structures = XRayGenerator.generate(curve_num, layer)
                    XRayGenerator.save(save_path + "/data", LAYERS_STR[layer], structures, noisy=noisy)
                else: #Generate data using neutron probe.
                    structures = NeutronGenerator.generate(curve_num, layer)
                    NeutronGenerator.save(save_path + "/data", LAYERS_STR[layer], structures, noisy=noisy)

                print(">>> Creating images for {}-layer curves".format(layer))
                save_path_layer = data_path_layer = save_path + "/data/{}".format(LAYERS_STR[layer])
                #Create images for the generated curves, ready for input to the classifier and regressors.
                generate_images(data_path_layer, save_path_layer, [layer], xray=xray, chunk_size=chunk_size, display_status=False)

            layers_paths = [save_path + "/data/{}".format(LAYERS_STR[layer]) for layer in layers]
            merge(save_path + "/data", layers_paths, display_status=False) #Merge the curves for each layer for classification.

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
                regress(data_path_layer, layer, save_path, epochs=regressor_epochs, show_plots=show_plots, xray=xray) #Train the regressor.
            else:
                print(">>> Loading {}-layer regressor".format(LAYERS_STR[layer]))
                load_path_layer = save_path + "/{}-layer-regressor/full_model.h5".format(LAYERS_STR[layer]) #Load an existing regressor.
                regress(data_path_layer, layer, load_path=load_path_layer, train=False, show_plots=show_plots, xray=xray)
            print()
            
def plot_objective_dual(objective1, objective2):
    """Creates a plot of two predictions on the same axis (for the paper).

    Args:
        objective1 (Objective): the first refnx objective object to plot.
        objective2 (Objective): the second refnx objective object to plot.

    """
    fig = plt.figure(figsize=[9,7], dpi=600)
    ax = fig.add_subplot(111)

    #Get the data, errors and predictions for each objective.
    y1, y_err1, model1 = objective1._data_transform(model=objective1.generative())
    y2, y_err2, model2 = objective2._data_transform(model=objective2.generative())
    
    # Add the data in a transformed fashion.
    ax.errorbar(objective1.data.x, y1, y_err1, label="Dataset1",
                color="blue", marker="o", ms=3, lw=0, elinewidth=1, capsize=1.5)
    ax.errorbar(objective2.data.x, y2/100, y_err2/100, label="$\mathregular{Dataset2\ (x10^{-2})}$", 
                color="green", marker="o", ms=3, lw=0, elinewidth=1, capsize=1.5)
    
    #Add the predictions
    ax.plot(objective1.data.x, model1, color="red", label="Prediction1", zorder=20)
    ax.plot(objective2.data.x, model2/100, color="black", label="$\mathregular{Prediction2\ (x10^{-2})}$", zorder=20)
    ax.set_xlim((0, 0.35))
    ax.set_ylim((5e-10, 1.2))

    plt.xlabel("$\mathregular{Q\ (Å^{-1})}$", fontsize=11, weight='bold')
    plt.ylabel('Reflectivity (arb.)',         fontsize=11, weight='bold')
    plt.yscale('log')
    plt.legend()

if __name__ == "__main__":
    save_path = './models/neutron'
    layers     = [1, 2, 3]
    curve_num  = 50000
    chunk_size = 100
    noisy            = True
    xray             = False
    show_plots       = False
    generate_data    = True
    train_classifier = True
    train_regressor  = True
    #Pipeline.setup(save_path, layers, curve_num, chunk_size, noisy, xray, show_plots, generate_data,
    #               train_classifier, train_regressor, classifer_epochs=50, regressor_epochs=50)

    load_path = "./models/neutron"
    data_path = "./data/real"
    classifier_path = load_path + "/classifier/full_model.h5"
    regressor_paths = {i: load_path + "/{}-layer-regressor/full_model.h5".format(LAYERS_STR[i]) for i in range(1, 4)}
    models = Pipeline.run(data_path, data_path, classifier_path, regressor_paths, fit=True, n_iter=100, xray=xray)
    
    #Make sure to set fit to false otherwise fits are plotted.
    plot_objective_dual(models['test_sample_1.dat'].objective, models['test_sample_2.dat'].objective)