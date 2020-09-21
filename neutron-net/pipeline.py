import os, glob, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Suppress TensorFlow warnings

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.utils  import Sequence

from refnx.dataset  import ReflectDataset
from refnx.reflect  import SLD, ReflectModel
from refnx.analysis import Objective, CurveFitter

from generate_refnx import CurveGenerator
from generate_data  import ImageGenerator, generate_images, DEPTH_BOUNDS, SLD_BOUNDS
from merge_data     import merge
from classification import classify
from regression     import regress

LAYERS_STR = {1: "one", 2: "two", 3: "three"}
DIMS = (300, 300)
CHANNELS = 1
tf.compat.v1.disable_eager_execution()

class DataLoaderClassification(Sequence):
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
        indexes = [list(self.labels.keys())[k] for k in indexes]
        images, targets = self.__data_generation(indexes)
        return images, targets

    def __data_generation(self, indexes):
        # 'Generates data containing batch_size samples'
        images  = np.empty((self.batch_size, *self.dim, self.channels))
        classes = np.empty((self.batch_size, 1), dtype=int)

        for i, np_image_filename in enumerate(indexes):
            images[i,]  = np.load(np_image_filename)
            classes[i,] = self.labels[np_image_filename]
        
        return images, classes

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.labels.keys()))


class DataLoaderRegression(Sequence):
    ''' Use Keras sequence to load image data from a dictionary '''
    def __init__(self, labels_dict, dim, channels):
        'Initialisation'
        self.labels_dict = labels_dict            
        self.dim         = dim                     # Image dimensions
        self.channels    = channels                # Image channels                   
        self.on_epoch_end()

    def __len__(self):
        'Denotes number of batches per epoch'
        # batch_size set as 1
        return int(np.floor(len(self.labels_dict.keys()) / 1))

    def __getitem__(self, index):
        'Generates one batch of data'
        indexes = self.indexes[index: (index + 1)]
        indexes = [list(self.labels_dict.keys())[k] for k in indexes]
        images  = self.__data_generation(indexes)
        return images

    def __data_generation(self, indexes):
        # 'Generates data containing batch_size samples'
        images = np.empty((1, *self.dim, self.channels))
        layers = np.empty((1, 1))

        for i, np_image_filename in enumerate(indexes):
            images[i,] = np.load(np_image_filename)
            layers[i,] = self.labels_dict[np_image_filename]["class"]

        return images, layers

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.labels_dict.keys()))


class KerasDropoutPredicter():
    """ Class that takes trained models and uses Dropout at test time to make Bayesian-like predictions"""
    def __init__(self, models):
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
        steps_done = 0
        all_out = []
        steps = len(sequence)
        output_generator = KerasDropoutPredicter.__iter_sequence_infinite(sequence)

        while steps_done < steps:
            # Yield the sample image, and the number of layers it is predicted to have
            x, y = next(output_generator)
            results = []

            for i in range(n_iter):
                # If one-layer
                if y[0][0] == 1:
                    result = self.f_1([x, 1])
                # else if two-layer
                elif y[0][0] == 2:
                    result = self.f_2([x, 1])
                # else if three-layer
                elif y[0][0] == 3:
                    result = self.f_3([x, 1])
                results.append(result)

            results = np.array(results)
            prediction, uncertainty = results.mean(axis=0), results.std(axis=0)
            outs = np.array([prediction, uncertainty])

            if not all_out:
                for out in outs:
                    all_out.append([])

            for i, out in enumerate(outs):
                all_out[i].append(out)
            
            steps_done+=1
        return [np.concatenate(out, axis=1) for out in all_out]
    
    @staticmethod
    def __iter_sequence_infinite(sequence):
        """Infinite iterator for sequence"""
        while True:
            for item in sequence:
                yield item


class Model():
    si_sld    = 2.047
    roughness = 2
    dq        = 2
    scale     = 1
    
    def __init__(self, file_path, layers, predicted_slds, predicted_depths):
        self.structure = SLD(0, name='Air')
        
        for i in range(layers):
            layer = SLD(predicted_slds[i], name='Layer {}'.format(i+1))(thick=predicted_depths[i], rough=Model.roughness)
            layer.sld.real.setp(bounds=SLD_BOUNDS, vary=True)
            layer.thick.setp(bounds=DEPTH_BOUNDS,  vary=True)
            self.structure = self.structure | layer

        si_substrate = SLD(Model.si_sld, name='Si Substrate')(thick=0, rough=Model.roughness)
        self.structure = self.structure | si_substrate
        
        data = ReflectDataset(file_path)
        self.model = ReflectModel(self.structure, scale=Model.scale, dq=Model.dq)
        self.objective = Objective(self.model, data)    
    
    def fit(self):
        fitter = CurveFitter(self.objective)
        fitter.fit('differential_evolution')
    
    def plot_SLD(self):
        plt.figure()
        plt.plot(*self.structure.sld_profile())
        plt.ylabel('SLD /$10^{-6} \AA^{-2}$')
        plt.xlabel('distance / $\AA$')
    
    def plot_reflectivity(self, qMin=0.005, qMax=0.3, points=1000):
        q = np.linspace(qMin, qMax, points)
        plt.figure()
        plt.plot(q, self.model(q))
        plt.xlabel('Q')
        plt.ylabel('Reflectivity')
        plt.yscale('log')
        
    def plot_objective(self):
        self.objective.plot()
        plt.xlabel('Q')
        plt.ylabel('Reflectivity')
        plt.yscale('log')


class Pipeline:
    @staticmethod
    def run(data_path, save_path, classifier_path, regressor_paths):
        dat_files = glob.glob(os.path.join(data_path, '*.dat'))
        
        layer_predictions, npy_image_filenames = Pipeline.__classify(dat_files, save_path, classifier_path)
        print("Predicted number of layers: {}\n".format(layer_predictions))
        
        #Currently hardcoded for 1 .dat file
        sld_predictions, depth_predictions, sld_errors, depth_errors = Pipeline.__regress(data_path, save_path, regressor_paths, layer_predictions, npy_image_filenames)
        for i in range(layer_predictions[0]):
            print("Predicted layer {0} - SLD: {1}| Depth: {2}".format(i+1, sld_predictions[0][i], depth_predictions[0][i]))

        model = Model(dat_files[0], layer_predictions[0], sld_predictions[0], depth_predictions[0])
        #model.plot_SLD()
        #model.plot_reflectivity()
        
        model.fit()
        model.plot_objective()
        
        return layer_predictions, sld_predictions, depth_predictions
      
    @staticmethod
    def __classify(data_path, save_path, classifier_path):
        print("-------------- Classification -------------")
        dat_files, npy_image_filenames = Pipeline.__dat_files_to_npy_images(data_path, save_path)
        class_labels = dict(zip(npy_image_filenames, np.zeros((len(npy_image_filenames), 1))))

        classifier_loader = DataLoaderClassification(class_labels, DIMS, CHANNELS, 1)
        classifier = load_model(classifier_path)
        return np.argmax(classifier.predict(classifier_loader, verbose=1), axis=1), npy_image_filenames

    @staticmethod
    def __regress(data_path, save_path, regressor_paths, layer_predictions, npy_image_filenames):
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
        kdp_predictions = kdp.predict(loader, n_iter=1)
        
        #Predictions given as [depth_1, depth_2, depth_3], [sld_1, sld_2, sld_3]
        depth_predictions = ImageGenerator.scale_to_range(kdp_predictions[0][0], (0, 1), DEPTH_BOUNDS)
        sld_predictions   = ImageGenerator.scale_to_range(kdp_predictions[0][1], (0, 1), SLD_BOUNDS)

        #Errors given as [depth_std_1, depth_std_2], [sld_std_1, sld_std_2]
        depth_errors = ImageGenerator.scale_to_range(kdp_predictions[1][0], (0, 1), DEPTH_BOUNDS)
        sld_errors   = ImageGenerator.scale_to_range(kdp_predictions[1][1], (0, 1), SLD_BOUNDS)
        
        return sld_predictions, depth_predictions, sld_errors, depth_errors
    
    @staticmethod
    def __dat_files_to_npy_images(dat_files, save_path):
        """Locate any .dat files in given data_path, create .npy images and save them in save_path"""
        if dat_files == []:
            sys.exit("No .dat files found in save path")
        image_filenames = Pipeline.__create_images_from_directory(dat_files, save_path)
        return dat_files, image_filenames
    
    @staticmethod
    def __create_images_from_directory(dat_files, save_path):
        """Take list of .dat files, create .npy images and save them in savepath"""
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
            img = ImageGenerator.image_process(sample)
            np.save(name, img)
    
        return image_files
    
    @staticmethod
    def __identify_header(path, n=5, th=0.9):
        """Parse the .dat file header to find out if there are headings or if it's empty"""
        df1 = pd.read_csv(path, header='infer', nrows=n)
        df2 = pd.read_csv(path, header=None, nrows=n)
        sim = (df1.dtypes.values == df2.dtypes.values).mean()
        return 'infer' if sim < th else None
    
    @staticmethod
    def setup(save_path, layers=[1,2,3], curve_num=5000, chunk_size=1000, generate_data=True,
              train_classifier=True, train_regressor=True, classifer_epochs=2, regressor_epochs=2):
        if generate_data:
            print("-------------- Data Generation ------------")
            for layer in layers:
                print(">>> Generating {}-layer curves".format(layer))
                structures = CurveGenerator.generate(curve_num, layer, sld_bounds=(-0.5,6), thick_bounds=(20,1000), substrate_SLD=2.047)
                CurveGenerator.save(save_path + "/data", LAYERS_STR[layer], structures)
                
                print(">>> Creating images for {}-layer curves".format(layer))
                save_path_layer = data_path_layer = save_path + "/data/{}".format(LAYERS_STR[layer])
                generate_images(data_path_layer, save_path_layer, [layer], chunk_size=chunk_size, display_status=False)
            
            layers_paths = [save_path + "/data/{}".format(LAYERS_STR[layer]) for layer in layers]
            merge(save_path + "/data", layers_paths)
        
        print("\n-------------- Classification -------------")
        if train_classifier:
            print(">>> Training classifier")
            classify(save_path + "/data/merged", save_path, train=True, epochs=classifer_epochs)
        else:
            print(">>> Loading classifier")
            load_path = save_path + "/classifier/full_model.h5"
            classify(save_path + "/data/merged", load_path=load_path, train=False)
        
        print("\n---------------- Regression ---------------")
        for layer in layers:
            data_path_layer = save_path + "/data/{}".format(LAYERS_STR[layer])
            if train_regressor:
                print(">>> Training {}-layer regressor".format(LAYERS_STR[layer]))
                regress(data_path_layer, layer, save_path, epochs=regressor_epochs)
            else:
                print(">>> Loading {}-layer regressor".format(LAYERS_STR[layer]))
                load_path_layer = save_path + "/{}-layer-regressor/full_model.h5".format(LAYERS_STR[layer])
                regress(data_path_layer, layer, load_path=load_path_layer, train=False)
            print()

if __name__ == "__main__":
    save_path = './models/investigate'
    layers     = [1, 2]
    curve_num  = 25000
    chunk_size = 1000
    generate_data    = True
    train_classifier = True
    train_regressor  = True
    #Pipeline.setup(save_path, layers, curve_num, chunk_size, generate_data, train_classifier, train_regressor, classifer_epochs=15, regressor_epochs=10)
    
    load_path = "./models/investigate"
    data_path = "./models/investigate"
    classifier_path = load_path + "/classifier/full_model.h5"
    regressor_paths = {1: load_path + "/one-layer-regressor/full_model.h5", 2: load_path + "/two-layer-regressor/full_model.h5"}
    Pipeline.run(data_path, save_path, classifier_path, regressor_paths)
