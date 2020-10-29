import h5py
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

from generate_data import ImageGenerator, LAYERS_STR, DIMS, CHANNELS
from regression    import DataLoader

class KerasDropoutPredicter():
    """KerasDropoutPredicter takes trained models and uses dropout at test time to make Bayesian-like predictions."""

    def __init__(self, model, sequence):
        """Initialises the dropout predictor with given model for a specific layer.

        Args:
            models (Model): Keras Model object.
            sequence (DataLoader): data to test the loaded model on.

        """
        # Define model with toggleable Dropout, K.learning_phase()
        self.f = K.function(
            [model.layers[0].input, K.learning_phase()],
            [model.layers[-2].output, model.layers[-1].output])

    def predict(self, sequencer, steps=None, n_iter=5, xray=False):
        """Makes Bayesian-like predictions using give model.

        Args:
            sequencer (DataLoader): the sequence providing data to predict on.
            steps (int): the number of batches of data to plot.
            n_iter (int): the number of iterations per step.
            xray (Boolean): whether input data is neutron or x-ray.

        Returns:
            List of numpy arrays of depth and SLD predictions, labels and associated errors.

        """
        steps_done = 0
        all_out = []
        if steps is None: #If the number of steps is not provided, plot the whole dataset.
            steps = len(sequencer)
        output_generator = KerasDropoutPredicter.__iter_sequence_infinite(sequencer)

        while steps_done < steps:
            generator_output = next(output_generator)
            images, targets = generator_output #Get a batch of images and targets.

            results = []
            for i in range(n_iter):
                [depths, slds] = self.f([images, 1]) #Set Dropout to True: 1
                #De-scale predictions
                depth_unscaled = ImageGenerator.scale_to_range(depths, (0, 1), ImageGenerator.depth_bounds)

                if xray:
                    sld_unscaled = ImageGenerator.scale_to_range(slds, (0, 1), ImageGenerator.sld_xray_bounds)
                else:
                    sld_unscaled = ImageGenerator.scale_to_range(slds, (0, 1), ImageGenerator.sld_neutron_bounds)

                results.append([depth_unscaled, sld_unscaled])

            results = np.array(results)
            prediction, uncertainty = results.mean(axis=0), results.std(axis=0)
            #De-scale targets
            targets_depth = ImageGenerator.scale_to_range(targets["depth"], (0, 1), ImageGenerator.depth_bounds)

            if xray:
                targets_sld = ImageGenerator.scale_to_range(targets["sld"], (0, 1), ImageGenerator.sld_xray_bounds)
            else:
                targets_sld = ImageGenerator.scale_to_range(targets["sld"], (0, 1), ImageGenerator.sld_neutron_bounds)

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
    """The Plotter class plots regression ground truths against predictions.

    Class Attributes:
        depth_axis (tuple): the range of depth values for the plot.
        sld_neutron_axis (tuple): the range of neutron SLD values for the plot.
        sld_xray_axis (tuple): the range of xray SLD values for the plot.
        depth_ticks (tuple): the values to place ticks on the depth axis.
        sld_neutron_ticks (tuple): the values to place ticks on the neutron SLD axis.
        sld_xray_ticks (tuple): the values to place ticks on the x-ray SLD axis.
        pad (int): padding value for the layer annotations.
        v_pad (int): vertical padding value for the layer annotations.

    """
    depth_axis       = (-250, 1250)
    sld_neutron_axis = (-1.5, 11)
    sld_xray_axis    = (0, 160)

    depth_ticks       = (0, 250, 500, 750, 1000)
    sld_neutron_ticks = (0, 2.5, 5, 7.5, 10)
    sld_xray_ticks    = (0, 30, 60, 90, 120, 150)

    pad   = 55
    v_pad = 80

    @staticmethod
    def __depth_subplot(ax, labels, preds, errors, x_axis_label=False):
        """Plots a ground truth against prediction plot for depths of a given layer.

        Args:
            ax (axis): the depth subplot to add to.
            labels (ndarray): ground truth values for depths.
            preds (ndarray): predidctions for depths.
            errors (ndarray): errors in predictions for depths.
            x_axis_label (Boolean): whehter to add the x-axis label or not.

        """
        ax.errorbar(labels, preds, errors, fmt="o", mec="k", mew=0.5, alpha=0.6, capsize=3, color="b", zorder=-130, markersize=4)
        ax.plot([0,1], [0,1], 'k', transform=ax.transAxes)
        ax.set_ylabel("$\mathregular{Depth_{predict}\ (Å)}$", fontsize=11, weight="bold")
        if x_axis_label:
            ax.set_xlabel("$\mathregular{Depth_{true}\ (Å)}$", fontsize=10, weight="bold")

    @staticmethod
    def __sld_subplot(ax, labels, preds, errors, x_axis_label=False, xray=False):
        """Plots a ground truth against prediction plot for SLDs of a given layer.

        Args:
            ax (axis): the SLD subplot to add to.
            labels (ndarray): ground truth values for SLDs.
            preds (ndarray): predictions for SLDs.
            errors (ndarray): errors in predictions for SLDs.
            x_axis_label (Boolean): whether to add the x-axis label or not.
            xray (Boolean): whether data to plot is x-ray or neutron.

        """
        ax.errorbar(labels, preds, errors, fmt="o", mec="k", mew=0.5, alpha=0.6, capsize=3, color="g", zorder=-130, markersize=4)
        ax.plot([0,1], [0,1], 'k', transform=ax.transAxes)
        if xray:
            ax.set_xlim(Plotter.sld_xray_axis)
            ax.set_ylim(Plotter.sld_xray_axis)
            ax.set_yticks(Plotter.sld_xray_ticks)
            ax.set_yticklabels(Plotter.sld_xray_ticks)
            ax.set_xticks(Plotter.sld_xray_ticks)
            ax.set_xticklabels(Plotter.sld_xray_ticks)
        else:
            ax.set_xlim(Plotter.sld_neutron_axis)
            ax.set_ylim(Plotter.sld_neutron_axis)
            ax.set_yticks(Plotter.sld_neutron_ticks)
            ax.set_yticklabels(Plotter.sld_neutron_ticks)
            ax.set_xticks(Plotter.sld_neutron_ticks)
            ax.set_xticklabels(Plotter.sld_neutron_ticks)

        ax.set_ylabel("$\mathregular{SLD_{predict}\ (Å^{-3})}$", fontsize=11, weight="bold")
        if x_axis_label:
            ax.set_xlabel("$\mathregular{SLD_{true}\ (Å^{-3})}$", fontsize=10, weight="bold")

    @staticmethod
    def __one_layer_plot(preds, labels, errors, xray=False):
        """Plots ground truths against predictions for depths and SLDs of 1-layer curves.

        Args:
            preds (ndarray): KDP predictions for depth and SLD.
            labels (ndarray): corresponding ground truths for depth and SLD predictions.
            errors (ndarray): errors in depth and SLD KDP predictions.
            xray (Boolean): whether data uses a x-ray or neutron probe.

        """
        fig, (ax_depth, ax_sld) = plt.subplots(1, 2, figsize=(8,4.5)) #Make the subplot structure for one layer.
        fig.subplots_adjust(wspace=0.35)
        fig.suptitle("One Layer - Predictions Against Ground Truths", size=16)

        #Add depth and SLD subplots for the single layer
        Plotter.__depth_subplot(ax_depth, labels[:,0], preds[:,0], errors[:,0], x_axis_label=True)
        Plotter.__sld_subplot(ax_sld,     labels[:,1], preds[:,1], errors[:,1], x_axis_label=True, xray=xray)
        ax_depth.annotate("Layer 1", xy=(0, 0.5),
                          xytext=(-ax_depth.yaxis.labelpad - Plotter.pad, Plotter.v_pad),
                          xycoords="axes points", textcoords="offset points",
                          size="large", ha="right", va="center")
        plt.show()

    @staticmethod
    def __two_layer_plot(preds, labels, errors, xray=False):
        """Plots ground truths against predictions for depths and SLDs of 2-layer curves.

        Args:
            preds (ndarray): KDP predictions for depth and SLD.
            labels (ndarray): corresponding ground truths for depth and SLD predictions.
            errors (ndarray): errors in depth and SLD KDP predictions.
            xray (Boolean): whether data uses a x-ray or neutron probe.

        """
        fig, axes = plt.subplots(2, 2, figsize=(8,8)) #Make the subplot structure for two layers.
        fig.subplots_adjust(wspace=0.30, top=0.94)
        fig.suptitle("Two Layer - Predictions Against Ground Truths", size=16)

        #Add depth and SLD subplots for the first layer
        Plotter.__depth_subplot(axes[0][0], labels[:,0], preds[:,0], errors[:,0])
        Plotter.__sld_subplot(axes[0][1],   labels[:,1], preds[:,1], errors[:,1], xray=xray)
        axes[0][0].annotate("Layer 1",
                            xy=(0, 0.5), xytext=(-axes[0][0].yaxis.labelpad - Plotter.pad, Plotter.v_pad),
                            xycoords="axes points", textcoords="offset points",
                            size="large", ha="right", va="center")

        #Add depth and SLD subplots for the second layer
        Plotter.__depth_subplot(axes[1][0], labels[:,2], preds[:,2], errors[:,2], x_axis_label=True)
        Plotter.__sld_subplot(axes[1][1],   labels[:,3], preds[:,3], errors[:,3], x_axis_label=True, xray=xray)
        axes[1][0].annotate("Layer 2", xy=(0, 0.5),
                            xytext=(-axes[1][0].yaxis.labelpad - Plotter.pad, Plotter.v_pad),
                            xycoords="axes points", textcoords="offset points",
                            size="large", ha="right", va="center")
        plt.show()

    @staticmethod
    def __three_layer_plot(preds, labels, errors, xray=False):
        """Plots ground truths against predictions for depths and SLDs of 3-layer curves.

        Args:
            preds (ndarray): KDP predictions for depth and SLD.
            labels (ndarray): corresponding ground truths for depth and SLD predictions.
            errors (ndarray): errors in depth and SLD KDP predictions.
            xray (Boolean): whether data uses a x-ray or neutron probe.

        """
        fig, axes = plt.subplots(3, 2, figsize=(8,9)) #Make the subplot structure for three layers.
        fig.subplots_adjust(wspace=0.30, hspace=0.2, top=0.94)
        fig.suptitle("Three Layer - Predictions Against Ground Truths", size=16)

        #Add depth and SLD subplots for the first layer
        Plotter.__depth_subplot(axes[0][0], labels[:,0], preds[:,0], errors[:,0])
        Plotter.__sld_subplot(axes[0][1],   labels[:,1], preds[:,1], errors[:,1], xray=xray)
        axes[0][0].annotate("Layer 1", xy=(0, 0.5),
                            xytext=(-axes[0][0].yaxis.labelpad - Plotter.pad, Plotter.v_pad),
                            xycoords="axes points", textcoords="offset points",
                            size="large", ha="right", va="center")

        #Add depth and SLD subplots for the second layer
        Plotter.__depth_subplot(axes[1][0], labels[:,2], preds[:,2], errors[:,2])
        Plotter.__sld_subplot(axes[1][1],   labels[:,3], preds[:,3], errors[:,3], xray=xray)
        axes[1][0].annotate("Layer 2", xy=(0, 0.5),
                            xytext=(-axes[1][0].yaxis.labelpad - Plotter.pad, Plotter.v_pad),
                            xycoords="axes points", textcoords="offset points",
                            size="large", ha="right", va="center")

        #Add depth and SLD subplots for the third layer
        Plotter.__depth_subplot(axes[2][0], labels[:,4], preds[:,4], errors[:,4], x_axis_label=True)
        Plotter.__sld_subplot(axes[2][1],   labels[:,5], preds[:,5], errors[:,5], x_axis_label=True, xray=xray)
        axes[2][0].annotate("Layer 3", xy=(0, 0.5),
                            xytext=(-axes[1][0].yaxis.labelpad - Plotter.pad, Plotter.v_pad),
                            xycoords="axes points", textcoords="offset points",
                            size="large", ha="right", va="center")
        plt.show()

    @staticmethod
    def kdp_plot(data_path, load_paths, steps=None, batch_size=20, n_iter=100, xray=False):
        """Performs dropout at test time to make Bayesian-like predictions and plots the results.

        Args:
            data_path (string): path to the directory containing 'test.h5' files to test with.
            load_paths (dict): a dictionary containing file paths to trained models for each layer.
            steps (int): the number of batches of data to plot.
            batch_size (int): the batch_size for loading data to test with.
            n_iter (int): the number of times to perform a prediction on a single curve (mean is taken of these).
            xray (Boolean): whether input data is neutron or x-ray.

        """
        layers = len(load_paths.values()) #Get the number of layers (either 2 or 3).
        for layer in range(1, layers+1): #Iterate over each layer to make separate plots for each.
            print(">>> Creating regression plot for {}-layer samples".format(LAYERS_STR[layer]))
            model  = load_model(load_paths[layer])
            file   = h5py.File(data_path+"/{}/test.h5".format(LAYERS_STR[layer]), "r")
            loader = DataLoader(file, DIMS, CHANNELS, batch_size, layer)
            kdp    = KerasDropoutPredicter(model, loader)

            preds = kdp.predict(loader, steps, n_iter, xray) #Perform KDP predictions and format results.
            depth, sld,              = preds[0][0], preds[0][1]
            depth_ground, sld_ground = preds[2][0], preds[2][1]
            depth_std, sld_std       = preds[1][0], preds[1][1]

            m = len(depth)
            preds_padded  = np.zeros((m, 2*layers)) #Pad results to ensure they match up with ground truths.
            errors_padded = np.zeros((m, 2*layers))
            labels_padded = np.zeros((m, 2*layers))
            for k in range(layer):
                preds_padded[:,2*k]   = depth[:,k]
                preds_padded[:,2*k+1] = sld[:,k]

                errors_padded[:,2*k]   = depth_std[:,k]
                errors_padded[:,2*k+1] = sld_std[:,k]

                labels_padded[:,2*k]   = depth_ground[:,k]
                labels_padded[:,2*k+1] = sld_ground[:,k]

            #Call the relevant method for each layer.
            if layer == 1:
                Plotter.__one_layer_plot(preds_padded, labels_padded, errors_padded, xray)
            elif layer == 2:
                Plotter.__two_layer_plot(preds_padded, labels_padded, errors_padded, xray)
            elif layer == 3:
                Plotter.__three_layer_plot(preds_padded, labels_padded, errors_padded, xray)


if __name__ == "__main__":
    layers = 3
    xray   = False
    data_path  = "./models/neutron/data"
    load_paths = {i: "./models/neutron/{}-layer-regressor/full_model.h5".format(LAYERS_STR[i]) for i in range(1, layers+1)}

    Plotter.kdp_plot(data_path, load_paths, steps=10, n_iter=100, xray=xray)
