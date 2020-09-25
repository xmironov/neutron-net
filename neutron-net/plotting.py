import h5py
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

from generate_data  import ImageGenerator, LAYERS_STR
from regression     import DataLoader
from classification import DIMS, CHANNELS

class KerasDropoutPredicter():
    """KerasDropoutPredicter takes trained models and uses dropout at test time to make Bayesian-like predictions."""
    
    def __init__(self, model, sequence):
        """Initialises the dropout predictor with given model for a specific layer.

        Args:
            models (Model): keras Model object.
            sequence (DataLoader): data to test the loaded model on.

        """
        # Define model with toggleable Dropout, K.learning_phase()
        self.f = K.function(
            [model.layers[0].input, K.learning_phase()], 
            [model.layers[-2].output, model.layers[-1].output])

    def predict(self, sequencer, n_iter=5):
        """Makes Bayesian-like predictions using give model.

        Args:
            sequence (DataLoader): the sequence providing data to predict on.
            n_iter (int): the number of iterations per step.

        Returns:
            List of numpy arrays of depth and SLD predictions, labels and associated errors.

        """
        steps_done = 0
        all_out = []
        steps = len(sequencer)
        output_generator = KerasDropoutPredicter.__iter_sequence_infinite(sequencer)

        while steps_done < steps:
            generator_output = next(output_generator)
            images, targets = generator_output

            results = []
            for i in range(n_iter):
                [depths, slds] = self.f([images, 1]) # Set Dropout to True: 1
                #De-scale predictions
                depth_scaled = ImageGenerator.scale_to_range(depths, (0, 1), ImageGenerator.depth_bounds)
                sld_scaled   = ImageGenerator.scale_to_range(slds,   (0, 1), ImageGenerator.sld_bounds)
                results.append([depth_scaled, sld_scaled])
            
            results = np.array(results)
            prediction, uncertainty = results.mean(axis=0), results.std(axis=0)
            #De-scale targets
            targets_depth = ImageGenerator.scale_to_range(targets["depth"], (0, 1), ImageGenerator.depth_bounds)
            targets_sld   = ImageGenerator.scale_to_range(targets["sld"],   (0, 1), ImageGenerator.sld_bounds)
            targets_sum = np.array([targets_depth, targets_sld])
            outs = np.array([prediction, uncertainty, targets_sum])

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


class Plotter:
    depth_axis  = (-250, 3250)
    sld_axis    = (-1, 11)
    dpeth_ticks = (0, 1000, 2000, 3000)
    sld_ticks   = (0, 2.5, 5, 7.5, 10)
    pad   = 55
    v_pad = 80
    
    @staticmethod
    def __depth_subplot(ax, labels, preds, errors, x_axis_label=False):
        ax.errorbar(labels, preds, errors, fmt="o", mec="k", mew=0.5, alpha=0.6, capsize=3, color="b", zorder=-130, markersize=4)
        ax.plot([0,1], [0,1], 'k', transform=ax.transAxes)
        ax.set_xlim(Plotter.depth_axis)
        ax.set_ylim(Plotter.depth_axis)
        ax.set_yticks(Plotter.depth_ticks)
        ax.set_yticklabels(Plotter.depth_ticks)
        ax.set_xticks(Plotter.depth_ticks)
        ax.set_xticklabels(Plotter.depth_ticks)
        ax.set_ylabel("$\mathregular{Depth_{predict}\ (Å)}$", fontsize=11, weight="bold")
        if x_axis_label:
            ax.set_xlabel("$\mathregular{Depth_{true}\ (Å)}$",    fontsize=10, weight="bold")
    
    @staticmethod
    def __sld_subplot(ax, labels, preds, errors, x_axis_label=False):
        ax.errorbar(labels, preds, errors, fmt="o", mec="k", mew=0.5, alpha=0.6, capsize=3, color="g", zorder=-130, markersize=4)
        ax.plot([0,1], [0,1], 'k', transform=ax.transAxes)
        ax.set_xlim(Plotter.sld_axis)
        ax.set_ylim(Plotter.sld_axis)
        ax.set_yticks(Plotter.sld_ticks)
        ax.set_yticklabels(Plotter.sld_ticks)
        ax.set_xticks(Plotter.sld_ticks)
        ax.set_xticklabels(Plotter.sld_ticks)
        ax.set_ylabel("$\mathregular{SLD_{predict}\ (Å^{-2})}$", fontsize=11, weight="bold")
        if x_axis_label:
            ax.set_xlabel("$\mathregular{SLD_{true}\ (Å^{-2})}$",    fontsize=10, weight="bold")
    
    @staticmethod
    def __one_layer_plot(preds, labels, errors):
        fig, (ax_depth, ax_sld) = plt.subplots(1, 2, figsize=(8,4.5))
        fig.subplots_adjust(wspace=0.35)
        fig.suptitle("One Layer - Predictions Against Ground Truths", size=16)
        
        Plotter.__depth_subplot(ax_depth, labels[:,0], preds[:,0], errors[:,0], x_axis_label=True)
        Plotter.__sld_subplot(ax_sld, labels[:,1], preds[:,1], errors[:,1], x_axis_label=True)
        ax_depth.annotate("Layer 1", xy=(0, 0.5), 
                          xytext=(-ax_depth.yaxis.labelpad - Plotter.pad, Plotter.v_pad),
                          xycoords="axes points", textcoords="offset points",
                          size="large", ha="right", va="center")
        plt.show()
    
    @staticmethod
    def __two_layer_plot(preds, labels, errors):    
        fig, axes = plt.subplots(2, 2, figsize=(8,8))
        fig.subplots_adjust(wspace=0.30, top=0.94)
        fig.suptitle("Two Layer - Predictions Against Ground Truths", size=16)
        
        #First layer
        Plotter.__depth_subplot(axes[0][0], labels[:,0], preds[:,0], errors[:,0])
        Plotter.__sld_subplot(axes[0][1], labels[:,1], preds[:,1], errors[:,1])
        axes[0][0].annotate("Layer 1", 
                            xy=(0, 0.5), xytext=(-axes[0][0].yaxis.labelpad - Plotter.pad, Plotter.v_pad),
                            xycoords="axes points", textcoords="offset points",
                            size="large", ha="right", va="center")
        
        #Second layer
        Plotter.__depth_subplot(axes[1][0], labels[:,2], preds[:,2], errors[:,2], x_axis_label=True)
        Plotter.__sld_subplot(axes[1][1], labels[:,3], preds[:,3], errors[:,3], x_axis_label=True)
        axes[1][0].annotate("Layer 2", xy=(0, 0.5), 
                            xytext=(-axes[1][0].yaxis.labelpad - Plotter.pad, Plotter.v_pad),
                            xycoords="axes points", textcoords="offset points",
                            size="large", ha="right", va="center")
        plt.show()
    
    @staticmethod
    def __three_layer_plot(preds, labels, errors):
        fig, axes = plt.subplots(3, 2, figsize=(8,9))
        fig.subplots_adjust(wspace=0.30, hspace=0.2, top=0.94)
        fig.suptitle("Three Layer - Predictions Against Ground Truths", size=16)
        
        #First layer
        Plotter.__depth_subplot(axes[0][0], labels[:,0], preds[:,0], errors[:,0])
        Plotter.__sld_subplot(axes[0][1], labels[:,1], preds[:,1], errors[:,1])
        axes[0][0].annotate("Layer 1", xy=(0, 0.5), 
                            xytext=(-axes[0][0].yaxis.labelpad - Plotter.pad, Plotter.v_pad),
                            xycoords="axes points", textcoords="offset points",
                            size="large", ha="right", va="center")
        
        #Second layer
        Plotter.__depth_subplot(axes[1][0], labels[:,2], preds[:,2], errors[:,2])
        Plotter.__sld_subplot(axes[1][1], labels[:,3], preds[:,3], errors[:,3])
        axes[1][0].annotate("Layer 2", xy=(0, 0.5), 
                            xytext=(-axes[1][0].yaxis.labelpad - Plotter.pad, Plotter.v_pad),
                            xycoords="axes points", textcoords="offset points",
                            size="large", ha="right", va="center")
        
        #Third layer
        Plotter.__depth_subplot(axes[2][0], labels[:,4], preds[:,4], errors[:,4], x_axis_label=True)
        Plotter.__sld_subplot(axes[2][1], labels[:,5], preds[:,5], errors[:,5], x_axis_label=True)
        axes[2][0].annotate("Layer 3", xy=(0, 0.5), 
                            xytext=(-axes[1][0].yaxis.labelpad - Plotter.pad, Plotter.v_pad),
                            xycoords="axes points", textcoords="offset points",
                            size="large", ha="right", va="center")
        plt.show()
    
    @staticmethod
    def kdp_plot(data_path, load_paths, batch_size=20, n_iter=2):
        layers = len(load_paths.values())
        for layer in range(1, layers+1):
            model  = load_model(load_paths[layer])
            file   = h5py.File(data_path+"/{}/test.h5".format(LAYERS_STR[layer]), "r")
            loader = DataLoader(file, DIMS, CHANNELS, batch_size, layer)
            kdp    = KerasDropoutPredicter(model, loader)
            
            preds = kdp.predict(loader, n_iter=n_iter)
            depth, sld,              = preds[0][0], preds[0][1]
            depth_ground, sld_ground = preds[2][0], preds[2][1]
            depth_std, sld_std       = preds[1][0], preds[1][1]
    
            m = len(depth)
            preds_padded  = np.zeros((m, 2*layers))
            errors_padded = np.zeros((m, 2*layers))
            labels_padded = np.zeros((m, 2*layers))
            for k in range(layer):
                preds_padded[:,2*k]   = depth[:,k]
                preds_padded[:,2*k+1] = sld[:,k]
                
                errors_padded[:,2*k]   = depth_std[:,k]
                errors_padded[:,2*k+1] = sld_std[:,k]
                
                labels_padded[:,2*k]   = depth_ground[:,k]
                labels_padded[:,2*k+1] = sld_ground[:,k]
    
            if layer == 1:
                Plotter.__one_layer_plot(preds_padded, labels_padded, errors_padded)
            elif layer == 2:
                Plotter.__two_layer_plot(preds_padded, labels_padded, errors_padded)
            elif layer == 3:
                Plotter.__three_layer_plot(preds_padded, labels_padded, errors_padded)
    

if __name__ == "__main__":
    layers = 3
    data_path  = "./models/investigate/data"
    load_paths = {i: "./models/investigate/{}-layer-regressor/full_model.h5".format(LAYERS_STR[i]) for i in range(1, layers+1)}
    Plotter.kdp_plot(data_path, load_paths)
