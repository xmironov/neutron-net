import h5py
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

from generate_data  import *
from regression     import DataLoader
from classification import DIMS, CHANNELS

DEPTH_AXIS  = (-250, 3250)
DEPTH_TICKS = (0, 1000, 2000, 3000)
SLD_AXIS  = (-1, 11)
SLD_TICKS = (0, 2.5, 5, 7.5, 10)
PAD = 55
V_PAD = 80

class KerasDropoutPredicter():
    def __init__(self, model, sequence):
        # Define model with toggleable Dropout, K.learning_phase()
        self.f = K.function(
            [model.layers[0].input, K.learning_phase()], 
            [model.layers[-2].output, model.layers[-1].output])

    def predict(self, sequencer, n_iter=5):
        steps_done = 0
        all_out = []
        steps = len(sequencer)
        output_generator = KerasDropoutPredicter.__iter_sequence_infinite(sequencer)

        while steps_done < steps:
            generator_output = next(output_generator)
            images, targets = generator_output

            results = []
            for i in range(n_iter):
                # Set Dropout to True: 1
                [depths, slds] = self.f([images, 1])
                depth_scaled = ImageGenerator.scale_to_range(depths, (0, 1), DEPTH_BOUNDS)
                sld_scaled   = ImageGenerator.scale_to_range(slds,   (0, 1), SLD_BOUNDS)
                results.append([depth_scaled, sld_scaled])
            
            results = np.array(results)
            prediction, uncertainty = results.mean(axis=0), results.std(axis=0)
            targets_depth = ImageGenerator.scale_to_range(targets["depth"], (0, 1), DEPTH_BOUNDS)
            targets_sld   = ImageGenerator.scale_to_range(targets["sld"],   (0, 1), SLD_BOUNDS)
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


class RegressionPlot:
    @staticmethod
    def __depth_subplot(ax, labels, preds, errors, x_axis_label=False):
        ax.errorbar(labels, preds, errors, fmt="o", mec="k", mew=0.5, alpha=0.6, capsize=3, color="b", zorder=-130, markersize=4)
        ax.plot([0,1], [0,1], 'k', transform=ax.transAxes)
        ax.set_xlim(DEPTH_AXIS)
        ax.set_ylim(DEPTH_AXIS)
        ax.set_yticks(DEPTH_TICKS)
        ax.set_yticklabels(DEPTH_TICKS)
        ax.set_xticks(DEPTH_TICKS)
        ax.set_xticklabels(DEPTH_TICKS)
        ax.set_ylabel("$\mathregular{Depth_{predict}\ (Å)}$", fontsize=11, weight="bold")
        if x_axis_label:
            ax.set_xlabel("$\mathregular{Depth_{true}\ (Å)}$",    fontsize=10, weight="bold")
    
    @staticmethod
    def __sld_subplot(ax, labels, preds, errors, x_axis_label=False):
        ax.errorbar(labels, preds, errors, fmt="o", mec="k", mew=0.5, alpha=0.6, capsize=3, color="g", zorder=-130, markersize=4)
        ax.plot([0,1], [0,1], 'k', transform=ax.transAxes)
        ax.set_xlim(SLD_AXIS)
        ax.set_ylim(SLD_AXIS)
        ax.set_yticks(SLD_TICKS)
        ax.set_yticklabels(SLD_TICKS)
        ax.set_xticks(SLD_TICKS)
        ax.set_xticklabels(SLD_TICKS)
        ax.set_ylabel("$\mathregular{SLD_{predict}\ (Å^{-2})}$", fontsize=11, weight="bold")
        if x_axis_label:
            ax.set_xlabel("$\mathregular{SLD_{true}\ (Å^{-2})}$",    fontsize=10, weight="bold")
    
    @staticmethod
    def __one_layer_plot(preds, labels, errors):
        fig, (ax_depth, ax_sld) = plt.subplots(1, 2, figsize=(8,4.5))
        fig.subplots_adjust(wspace=0.35)
        fig.suptitle("One Layer - Predictions Against Ground Truths", size=16)
        
        ax_depth.annotate("Layer 1", xy=(0, 0.5), xytext=(-ax_depth.yaxis.labelpad - PAD, V_PAD),
                    xycoords="axes points", textcoords="offset points",
                    size="large", ha="right", va="center")
        
        RegressionPlot.__depth_subplot(ax_depth, labels[:,0], preds[:,0], errors[:,0], x_axis_label=True)
        RegressionPlot.__sld_subplot(ax_sld, labels[:,1], preds[:,1], errors[:,1], x_axis_label=True)
        plt.show()
    
    @staticmethod
    def __two_layer_plot(preds, labels, errors):    
        fig, axes = plt.subplots(2, 2, figsize=(8,8))
        fig.subplots_adjust(wspace=0.30, top=0.94)
        fig.suptitle("Two Layer - Predictions Against Ground Truths", size=16)
        
        #First layer
        RegressionPlot.__depth_subplot(axes[0][0], labels[:,0], preds[:,0], errors[:,0])
        axes[0][0].annotate("Layer 1", xy=(0, 0.5), xytext=(-axes[0][0].yaxis.labelpad - PAD, V_PAD),
                        xycoords="axes points", textcoords="offset points",
                        size="large", ha="right", va="center")
        RegressionPlot.__sld_subplot(axes[0][1], labels[:,1], preds[:,1], errors[:,1])
        
        #Second layer
        RegressionPlot.__depth_subplot(axes[1][0], labels[:,2], preds[:,2], errors[:,2], x_axis_label=True)
        axes[1][0].annotate("Layer 2", xy=(0, 0.5), xytext=(-axes[1][0].yaxis.labelpad - PAD, V_PAD),
                    xycoords="axes points", textcoords="offset points",
                    size="large", ha="right", va="center")
        RegressionPlot.__sld_subplot(axes[1][1], labels[:,3], preds[:,3], errors[:,3], x_axis_label=True)
        plt.show()
    
    @staticmethod
    def __three_layer_plot(preds, labels, errors):
        fig, axes = plt.subplots(3, 2, figsize=(8,9))
        fig.subplots_adjust(wspace=0.30, hspace=0.2, top=0.94)
        fig.suptitle("Three Layer - Predictions Against Ground Truths", size=16)
        
        #First layer
        RegressionPlot.__depth_subplot(axes[0][0], labels[:,0], preds[:,0], errors[:,0])
        axes[0][0].annotate("Layer 1", xy=(0, 0.5), xytext=(-axes[0][0].yaxis.labelpad - PAD, V_PAD),
                        xycoords="axes points", textcoords="offset points",
                        size="large", ha="right", va="center")
        RegressionPlot.__sld_subplot(axes[0][1], labels[:,1], preds[:,1], errors[:,1])
        
        #Second layer
        RegressionPlot.__depth_subplot(axes[1][0], labels[:,2], preds[:,2], errors[:,2])
        axes[1][0].annotate("Layer 2", xy=(0, 0.5), xytext=(-axes[1][0].yaxis.labelpad - PAD, V_PAD),
                    xycoords="axes points", textcoords="offset points",
                    size="large", ha="right", va="center")
        RegressionPlot.__sld_subplot(axes[1][1], labels[:,3], preds[:,3], errors[:,3])
        
        #Third layer
        RegressionPlot.__depth_subplot(axes[2][0], labels[:,4], preds[:,4], errors[:,4], x_axis_label=True)
        axes[2][0].annotate("Layer 3", xy=(0, 0.5), xytext=(-axes[1][0].yaxis.labelpad - PAD, V_PAD),
                    xycoords="axes points", textcoords="offset points",
                    size="large", ha="right", va="center")
        RegressionPlot.__sld_subplot(axes[2][1], labels[:,5], preds[:,5], errors[:,5], x_axis_label=True)
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
                RegressionPlot.__one_layer_plot(preds_padded, labels_padded, errors_padded)
            elif layer == 2:
                RegressionPlot.__two_layer_plot(preds_padded, labels_padded, errors_padded)
            elif layer == 3:
                RegressionPlot.__three_layer_plot(preds_padded, labels_padded, errors_padded)
    

if __name__ == "__main__":
    layers = 3
    data_path  = "./models/investigate/data"
    load_paths = {i: "./models/investigate/{}-layer-regressor/full_model.h5".format(LAYERS_STR[i]) for i in range(1, layers+1)}
    RegressionPlot.kdp_plot(data_path, load_paths)
