import os.path, glob
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from refnx.reflect import SLD, ReflectModel

from generate_refnx import CurveGenerator, NeutronGenerator
from generate_data  import DIMS, CHANNELS
from pipeline       import Pipeline, DataLoaderRegression, KerasDropoutPredicter

class TimeVarying:
    """The TimeVarying class contains all the code for generating and predicting on
       a dataset whose layer-one depth changes over time"""
    
    def __init__(self, path):
        """Initialises a TimeVarying object with given file path to save to or load from.
        
        Args:
            save_path (string): the file path to save or load datasets.
        """
        if not os.path.exists(path): #Create directory if not present.
            os.makedirs(path)
            
        self.path         = path
        self.points       = 500
        self.roughness    = 8
        self.layer1_sld   = 2.5
        self.layer2_sld   = 5.0
        self.layer2_thick = 100
        self.thick_min    = 100
        self.thick_max    = 900
        self.thick_step   = 50
        #Range of thicknesses over the experiment duration.
        self.thick_range  = np.arange(self.thick_min, self.thick_max, self.thick_step)
    
    def generate(self):
        """Generates a series of datasets simulating an experiment with a layer whose
           thickness changes over time."""
           
        print("---------------- Generating ----------------")
        
        q = np.logspace(np.log10(CurveGenerator.qMin), np.log10(NeutronGenerator.qMax), self.points)
        for thickness in self.thick_range: #Iterate over each thickness the top layer will take.
            #The structure consists of air followed by each layer and then finally the substrate.
            air       = SLD(0, name="Air")
            layer1    = SLD(self.layer1_sld, name="Layer 1")(thick=thickness, rough=self.roughness)
            layer2    = SLD(self.layer2_sld, name="Layer 2")(thick=self.layer2_thick, rough=self.roughness)
            substrate = SLD(NeutronGenerator.substrate_sld, name="Si Substrate")(thick=0, rough=self.roughness)
    
            structure = air | layer1 | layer2 | substrate
            model = ReflectModel(structure, bkg=NeutronGenerator.bkg, scale=CurveGenerator.scale, dq=CurveGenerator.dq)
            r = model(q)
    
            #Add simulated noise to the data.
            r_noisy_bkg    = CurveGenerator.background_noise(r, bkg_rate=CurveGenerator.bkg_rate)
            r_noisy_sample = CurveGenerator.sample_noise(q, r_noisy_bkg, constant=CurveGenerator.noise_constant)
    
            #CurveGenerator.plot_reflectivity(q, r_noisy_sample)        
            
            data = np.zeros((self.points, 3))
            data[:, 0] = q
            data[:, 1] = r_noisy_sample
            data[:, 2] = 1e-10 #Error is set to be (near) zero as it is not used by the networks. This could be improved.
            np.savetxt(self.path+"/{}.dat".format(thickness), data, delimiter="    ")
    
        print("Layer 1 - Depth: [{0},{1}] | SLD: {2:1.3f}".format(self.thick_min, self.thick_max-self.thick_step, self.layer1_sld))
        print("Layer 2 - Depth: {0}       | SLD: {1:1.3f}".format(self.layer2_thick, self.layer2_sld))

    def predict(self, regressor_path):
        """Performs predictions on generated time-varying data.
        
        Args:
            regressor_path (string): file path to the two-layer-regressor for prediction.
        
        """
        print("\n---------------- Predicting ----------------")
        dat_files = glob.glob(os.path.join(self.path, '*.dat')) #Search for .dat files.
        npy_image_filenames = Pipeline.dat_files_to_npy_images(dat_files, self.path) #Generate images for .dat files.

        values_labels = {}
        for filename in npy_image_filenames: #All labels are 2-layer.
            values_labels[filename] = {"depth": np.zeros((1, 2)), "sld": np.zeros((1, 2)), "class": 2}

        loader = DataLoaderRegression(values_labels, DIMS, CHANNELS)
        regressor = {2: load_model(regressor_path+"/full_model.h5")} #Only need to load a two-layer regressor.

        #Use custom class to activate Dropout at test time in models
        kdp = KerasDropoutPredicter(regressor)
        kdp_predictions = kdp.predict(loader, n_iter=100, xray=False) #Predict 100 times per .dat file

        depth_predictions = kdp_predictions[0][0]
        sld_predictions   = kdp_predictions[0][1]

        depth_errors = kdp_predictions[1][0]
        sld_errors   = kdp_predictions[1][1]

        #Print the predictions and errors for the depths and SLDs for each layer for each file.
        for curve in range(len(dat_files)): #Iterate over each file.
            filename = os.path.basename(dat_files[curve])
            print("Results for '{}'".format(filename))
            for i in range(2): #Iterate over each layer
                print(">>> Predicted layer {0} - SLD:   {1:10.4f} | Error: {2:10.6f}".format(i+1, sld_predictions[curve][i], sld_errors[curve][i]))
                print(">>> Predicted layer {0} - Depth: {1:10.4f} | Error: {2:10.6f}".format(i+1, depth_predictions[curve][i], depth_errors[curve][i]))
            print()
        
        self.__plot(depth_predictions, sld_predictions, depth_errors, sld_errors) #Generate plots for results.
      
    def __plot(self, depth_predictions, sld_predictions, depth_errors, sld_errors):
        """Creates a series of plots of predictions against ground-truths for generated time-varying data.
        
        Args:
            depth_predictions (ndarray): KDP predictions for layer one and two depths.
            sld_predictions (ndarray): KDP predictions for layer one and two SLDS.
            depth_errors (ndarray): KDP errors for layer one and two depth predictions.
            sld_errors (ndarray): KDP errors for layer one and two SLD predictions.
        
        """
        steps = len(self.thick_range)
        time_steps = np.arange(1, steps+1, 1) #The time steps of arbitrary units.
        
        #Plot the ground truth and predictions for both of the layers' SLDs and depths.
        depth_true = (self.thick_range,        [self.layer2_thick]*steps)
        sld_true   = ([self.layer1_sld]*steps, [self.layer2_sld]*steps)
        
        self.__plot_data(time_steps, depth_true, depth_predictions, depth_errors, "Depth", True)   
        self.__plot_data(time_steps, sld_true,   sld_predictions,   sld_errors,   "SLD", False) 
            
    def __plot_data(self, time_steps, ground_truths, predictions, errors, parameter, legend):
        """Creates an individual of plot of predictions against ground-truth.
        
        Args:
            time_steps (ndarray): the x-axis for the plot (how many datasets are the in the experiment).
            ground_truth (ndarray): the ground truth labels associated with given predictions
            predictions (ndarray): predictions for either depth or SLD made by the KDP.
            errors (ndarray): errors in the given predictions.
            parameter (string): either 'Depth' or 'SLD'.
            legend (Boolean): whether to display the legend or not.
        
        """
        true_layer1,  true_layer2  = ground_truths[0], ground_truths[1]
        pred_layer1,  pred_layer2  = predictions[:,0], predictions[:,1]
        error_layer1, error_layer2 = errors[:,0],      errors[:,1]
        
        fig = plt.figure(figsize=(8,5), dpi=600)
        ax = fig.add_subplot(111)
        ax.plot(time_steps, true_layer1, mew=0.5, alpha=0.6, c='b', label='Layer 1 Ground Truth')
        ax.plot(time_steps, true_layer2, mew=0.5, alpha=0.6, c='g', label='Layer 2 Ground Truth')
        ax.errorbar(time_steps, pred_layer1, error_layer1, fmt="o", mec="k", mew=0.5, alpha=0.6, capsize=3, color="b", zorder=-130, markersize=4, label='Layer 1 Prediction')
        ax.errorbar(time_steps, pred_layer2, error_layer2, fmt="o", mec="k", mew=0.5, alpha=0.6, capsize=3, color="g", zorder=-130, markersize=4, label='Layer 2 Prediction')
        ax.set_xlabel("Time Step", fontsize=10, weight="bold")
        
        if parameter == "SLD":
            #ax.set_ylim(*NeutronGenerator.sld_bounds)
            ax.set_ylabel("$\mathregular{SLD\ (x10^{-6} Å^{-2})}$", fontsize=10, weight="bold")
        elif parameter == "Depth":
            ax.set_ylim(*CurveGenerator.thick_bounds)
            ax.set_ylabel("$\mathregular{Depth\ (Å)}$", fontsize=10, weight="bold")
            
        if legend:
            plt.legend(loc='upper right')
        plt.show()


if __name__ == "__main__":
    save_path      = './data/time-varying'
    regressor_path = './models/neutron/two-layer-regressor'
    
    model = TimeVarying(save_path)
    model.generate()
    model.predict(regressor_path)
