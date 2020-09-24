import h5py
import numpy as np
import matplotlib.pyplot   as plt
import matplotlib.gridspec as gridspec

from tensorflow.keras        import backend as K
from tensorflow.keras.models import load_model

from regression     import DataLoader
from generate_data  import ImageGenerator, DEPTH_BOUNDS, SLD_BOUNDS

DIMS = (300, 300)
CHANNELS = 1

class KerasDropoutPredicter():
    def __init__(self, model, sequence):
        # Define model with toggleable Dropout, K.learning_phase()
        self.f = K.function(
            [model.layers[0].input, K.learning_phase()], 
            [model.layers[-2].output, model.layers[-1].output])

    def predict(self, sequencer, n_iter=2):
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
                result = self.f([images, 1])
                results.append(result)

            results = np.array(results)
            prediction, uncertainty = results.mean(axis=0), results.std(axis=0)
            outs = [prediction, uncertainty]
            targets_depth = targets["depth"]
            targets_sld = targets["sld"]
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


def plot(preds, labels, batch_size, error):
    # The following is specific for the graphs generated in the paper graphics
    labels_1_a = labels[0][:,0]
    preds_1_a  = preds[0][:,0]
    error_1_a  = error[0][:,0]

    labels_1_b = labels[0][:,1]
    preds_1_b  = preds[0][:,1]
    error_1_b  = error[0][:,1]

    labels_2_a = labels[1][:,0]
    preds_2_a  = preds[1][:,0]
    error_2_a  = error[1][:,0]

    labels_2_b = labels[1][:,1]
    preds_2_b  = preds[1][:,1]
    error_2_b  = error[1][:,1]

    labels_2_c = labels[1][:,2]
    preds_2_c  = preds[1][:,2]
    error_2_c  = error[1][:,2]

    labels_2_d = labels[1][:,3]
    preds_2_d  = preds[1][:,3]
    error_2_d  = error[1][:,3]

    remainder = len(labels) % batch_size
    if remainder:
        labels = labels[:-remainder]

    outer_grid = gridspec.GridSpec(3, 2, hspace=0.00, wspace=0.390, left=0.19, right=0.9, top=0.950, bottom=0.110)

    fig = plt.figure()
    pad = 55
    v_pad = 80

    depth_axis  = [-250, 3250]
    depth_ticks = [0, 1000, 2000, 3000]
    
    sld_axis = [-1, 11]
    sld_ticks_y = [0, 5, 10]
    sld_ticks_x = [0, 2.5, 5, 7.5, 10]

    ax0 = fig.add_subplot(outer_grid[0, 0])
    ax0.errorbar(labels_1_a, preds_1_a, error_1_a, fmt="o",mec="k",mew=.5,alpha=.6,capsize=3,color="b",zorder=-130,markersize=4)
    ax0.plot([0,1], [0,1], 'k', transform=ax0.transAxes)
    ax0.set_xlim(depth_axis)
    ax0.set_ylim(depth_axis)
    ax0.set_yticks(depth_ticks)
    ax0.set_yticklabels(depth_ticks)
    ax0.set_xticklabels([])
    ax0.annotate("$\mathregular{1^{i}}$", xy=(0., 0.5), xytext=(-ax0.yaxis.labelpad - pad, v_pad),
                        xycoords="axes points", textcoords="offset points",
                        size="large", ha="right", va="center")

    ax0.set_facecolor("xkcd:very light blue")

    ax0_1 = fig.add_subplot(outer_grid[0, 1])
    ax0_1.errorbar(labels_1_b, preds_1_b, error_1_b,fmt="o",mec="k",mew=.5,alpha=.6,capsize=3,color="g",zorder=-130,markersize=4)
    ax0_1.plot([0,1], [0,1], 'k', transform=ax0_1.transAxes)
    ax0_1.set_xlim(sld_axis)
    ax0_1.set_ylim(sld_axis)
    ax0_1.set_yticks(sld_ticks_y)
    ax0_1.set_yticklabels(sld_ticks_y)
    ax0_1.set_xticklabels([])
    ax0_1.set_facecolor("xkcd:very light blue")

    ax1 = fig.add_subplot(outer_grid[1, 0])
    ax1.errorbar(labels_2_a, preds_2_a, error_2_a,fmt="o",mec="k",mew=.5,alpha=.6,capsize=3,color="b",zorder=-130,markersize=4)
    ax1.plot([0,1], [0,1], 'k', transform=ax1.transAxes)
    ax1.set_xlim(depth_axis)
    ax1.set_ylim(depth_axis)
    ax1.set_yticks(depth_ticks)
    ax1.set_yticklabels(depth_ticks)
    ax1.set_xticklabels([])
    ax1.set_ylabel("$\mathregular{Depth_{predict}\ (Å)}$", fontsize=11, weight="bold")
    ax1.annotate("$\mathregular{2^{i}}$", xy=(0, 0.5), xytext=(-ax0.yaxis.labelpad - pad, v_pad),
                        xycoords="axes points", textcoords="offset points",
                        size="large", ha="right", va="center")

    ax2 = fig.add_subplot(outer_grid[1, 1])
    ax2.errorbar(labels_2_b, preds_2_b, error_2_b,fmt="o",mec="k",mew=.5,alpha=.6,capsize=3,color="g",zorder=-130,markersize=4)
    ax2.plot([0,1], [0,1], 'k', transform=ax2.transAxes)
    ax2.set_xlim(sld_axis)
    ax2.set_ylim(sld_axis)
    ax2.set_yticks(sld_ticks_y)
    ax2.set_yticklabels(sld_ticks_y)
    ax2.set_xticklabels([])
    ax2.set_ylabel("$\mathregular{SLD_{predict}\ (Å^{-2})}$", fontsize=11, weight="bold")

    ax3 = fig.add_subplot(outer_grid[2, 0])
    ax3.errorbar(labels_2_c, preds_2_c, error_2_c,fmt="o",mec="k",mew=.5,alpha=.6,capsize=3,color="b",zorder=-130,markersize=4)
    ax3.plot([0,1], [0,1], 'k', transform=ax3.transAxes)
    ax3.set_xlim(depth_axis)
    ax3.set_ylim(depth_axis)
    ax3.set_yticks(depth_ticks)
    ax3.set_yticklabels(depth_ticks)
    ax3.set_xlabel("$\mathregular{Depth_{true}\ (Å)}$", fontsize=10, weight="bold")
    ax3.annotate("$\mathregular{2^{ii}}$", xy=(0, 0.5), xytext=(-ax3.yaxis.labelpad - pad, v_pad),
                        xycoords="axes points", textcoords="offset points",
                        size="large", ha="right", va="center")

    ax4 = fig.add_subplot(outer_grid[2, 1])
    ax4.errorbar(labels_2_d, preds_2_d, error_2_d,fmt="o",mec="k",mew=.5,alpha=.6,capsize=3,color="g",zorder=-130,markersize=4)
    ax4.plot([0,1], [0,1], 'k', transform=ax4.transAxes)
    ax4.set_xlim(sld_axis)
    ax4.set_xticks(sld_ticks_x)
    ax4.set_xticklabels(sld_ticks_x)
    ax4.set_ylim(sld_axis)
    ax4.set_yticks(sld_ticks_y)
    ax4.set_yticklabels(sld_ticks_y)
    ax4.set_xlabel("$\mathregular{SLD_{true}\ (Å^{-2})}$", fontsize=10, weight="bold")

    plt.show()

def run_kdp(path_one_layer, path_two_layer, data_path, batch_size=20):
    model_one_layer = load_model(path_one_layer)
    model_two_layer = load_model(path_two_layer)
    
    test_one_layer = h5py.File(data_path+"/one/test.h5", "r")
    test_two_layer = h5py.File(data_path+"/two/test.h5", "r")

    loader_one_layer = DataLoader(test_one_layer, DIMS, CHANNELS, batch_size, 1)
    loader_two_layer = DataLoader(test_two_layer, DIMS, CHANNELS, batch_size, 2)
    
    kdp_one_layer = KerasDropoutPredicter(model_one_layer, loader_one_layer)
    kdp_two_layer = KerasDropoutPredicter(model_two_layer, loader_two_layer)

    preds_one_layer = kdp_one_layer.predict(loader_one_layer, n_iter=1)
    preds_two_layer = kdp_two_layer.predict(loader_two_layer, n_iter=1)

    ####################################################################################
    # y_predictions[n] : n = 0:mean prediction, 1:standard deviation, 2:ground truth
    # y_predictions[n][j] : j = 0:depth, 1:sld
    # y_predictions[n][j][k] : k = sample number
    # y_predictions[n][j][k][l] : l = 0:layer1, 1:layer2
    ###################################################################################
    
    depth_one = ImageGenerator.scale_to_range(preds_one_layer[0][0], (0, 1), DEPTH_BOUNDS)
    depth_two = ImageGenerator.scale_to_range(preds_two_layer[0][0], (0, 1), DEPTH_BOUNDS)
    sld_one   = ImageGenerator.scale_to_range(preds_one_layer[0][1], (0, 1), SLD_BOUNDS)
    sld_two   = ImageGenerator.scale_to_range(preds_two_layer[0][1], (0, 1), SLD_BOUNDS)
    
    depth_std_one = ImageGenerator.scale_to_range(preds_one_layer[1][0], (0, 1), DEPTH_BOUNDS)
    depth_std_two = ImageGenerator.scale_to_range(preds_two_layer[1][0], (0, 1), DEPTH_BOUNDS)
    sld_std_one   = ImageGenerator.scale_to_range(preds_one_layer[1][1], (0, 1), SLD_BOUNDS)
    sld_std_two   = ImageGenerator.scale_to_range(preds_two_layer[1][1], (0, 1), SLD_BOUNDS)

    depth_ground_one, sld_ground_one = preds_one_layer[2][0], preds_one_layer[2][1]
    depth_ground_two, sld_ground_two = preds_two_layer[2][0], preds_two_layer[2][1]

    preds_one  = np.c_[depth_one[:,0], sld_one[:,0], np.zeros(len(depth_one)), np.zeros(len(sld_one))]
    error_one  = np.c_[depth_std_one[:,0], sld_std_one[:,0], np.zeros(len(depth_one)), np.zeros(len(sld_one))]
    labels_one = np.c_[depth_ground_one[:,0], sld_ground_one[:,0], np.zeros(len(depth_one)), np.zeros(len(sld_one))]

    preds_two  = np.c_[depth_two[:,0], sld_two[:,0], depth_two[:,1], sld_two[:,1]]
    error_two  = np.c_[depth_std_two[:,0], sld_std_two[:,0], depth_std_two[:,1], sld_std_two[:,1]]
    labels_two = np.c_[depth_ground_two[:,0], sld_ground_two[:,0], depth_ground_two[:,1], sld_ground_two[:,1]]

    preds  = [preds_one,  preds_two]
    labels = [labels_one, labels_two]
    error  = [error_one,  error_two]
    plot(preds, labels, batch_size, error)


if __name__ == "__main__":
    path_one_layer = "./models/investigate/one-layer-regressor/full_model.h5"
    path_two_layer = "./models/investigate/two-layer-regressor/full_model.h5"
    data_path      = "./models/investigate/data"
    
    run_kdp(path_one_layer, path_two_layer, data_path)
