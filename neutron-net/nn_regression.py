from comet_ml import Experiment

import os 
os.environ["KMP_AFFINITY"] = "none"

import argparse
import h5py, time, re, sys 
import json, glob, pickle, random
import warnings

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from datetime import datetime
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, LeakyReLU, BatchNormalization, Dropout, Input
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, Nadam

DIMS = (300, 300)
CHANNELS = 1
tf.compat.v1.disable_eager_execution()

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
        output_generator = iter_sequence_infinite(sequencer)

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

class Net():
    def __init__(self, dims, channels, epochs, dropout, learning_rate, workers, layers, batch_size):
        'Initialisation'
        self.outputs       = layers
        self.dims          = dims
        self.channels      = channels
        self.epochs        = epochs
        self.dropout       = dropout
        self.learning_rate = learning_rate
        self.workers       = workers
        self.batch_size    = batch_size
        self.model         = self.create_model()

    def train(self, train_seq, valid_seq):
        'Trains data on Sequences'

        learning_rate_reduction_cbk = ReduceLROnPlateau(
            monitor='val_loss',
            patience=10,
            verbose=1,
            factor=0.5,
            min_lr = 0.000001
        )

        model_checkpoint_cbk = ModelCheckpoint(
            'weights.{epoch:02d}-{val_loss:2f}.h5',
            monitor='val_loss',
            verbose=0,
            save_best_only=True
        )

        start = time.time()
        self.history = self.model.fit(
            train_seq,
            validation_data = valid_seq,
            epochs = self.epochs,
            workers = self.workers,
            use_multiprocessing = False,
            verbose = 1,
            callbacks = [learning_rate_reduction_cbk]
        )

        elapsed_time = time.time() - start 
        self.time_taken = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

        train_seq.close_file()
        valid_seq.close_file()

        return self.history
    
    def test(self, test_seq, datapath):
        scaler = pickle.load(open(os.path.join(datapath, "output_scaler.p"), "rb"))
        preds = self.model.predict(test_seq, use_multiprocessing=False, verbose=1)
        depth, sld = preds[0], preds[1]

        if self.outputs == 2:
            padded_preds = np.c_[depth[:,0], sld[:,0], depth[:,1], sld[:,1]]
            self.preds = scaler.inverse_transform(padded_preds)

        elif self.outputs ==1:
            padded_preds = np.c_[depth[:,0], sld[:,0], np.zeros(len(depth)), np.zeros(len(sld))]
            self.preds = scaler.inverse_transform(padded_preds)

    def create_model(self):
        # Convolutional Encoder
        input_img = Input(shape=(*self.dims, self.channels))
        conv_1 = Conv2D(32, (3,3), activation='relu')(input_img)
        pool_1 = MaxPooling2D((2,2))(conv_1)
        conv_2 = Conv2D(64, (3,3), activation='relu')(pool_1)
        pool_2 = MaxPooling2D((2,2), strides=(2,2))(conv_2)
        conv_3 = Conv2D(32, (3,3), activation='relu')(pool_2)
        pool_3 = MaxPooling2D((2,2))(conv_3)
        conv_4 = Conv2D(16, (3,3), activation='relu')(pool_3)
        pool_4 = MaxPooling2D((2,2))(conv_4)
        flatten = Flatten()(pool_4)

        # Depth feed-forward
        dense_1_d = Dense(units=300, activation='relu', kernel_initializer='he_normal')(flatten)
        dropout_1_d = Dropout(self.dropout)(dense_1_d)
        dense_2_d = Dense(units=192, activation='relu', kernel_initializer='he_normal')(dropout_1_d)
        dropout_2_d = Dropout(self.dropout)(dense_2_d)
        dense_3_d = Dense(units=123, activation='relu', kernel_initializer='he_normal')(dropout_2_d)
        dropout_3_d = Dropout(self.dropout)(dense_3_d)
        dense_4_d = Dense(units=79, activation='relu', kernel_initializer='he_normal')(dropout_3_d)
        dropout_4_d = Dropout(self.dropout)(dense_4_d)
        dense_5_d = Dense(units=50, activation='relu', kernel_initializer='he_normal')(dropout_4_d)
        dropout_5_d = Dropout(self.dropout)(dense_5_d)
        depth_linear = Dense(units=self.outputs, activation='linear', name='depth')(dropout_5_d)
        sld_linear = Dense(units=self.outputs, activation='linear', name='sld')(dropout_5_d)

        model = Model(inputs=input_img, outputs=[depth_linear, sld_linear])
        model.compile(loss={'depth':'mse','sld':'mse'},
                        loss_weights={'depth':1,'sld':1},
                        optimizer = Nadam(self.learning_rate),
                        metrics={'depth':'mae','sld':'mae'})
        return model

    def summary(self):
        self.model.summary()

    def plot(self, labels, savepath):
        remainder = len(labels) % self.batch_size

        if remainder:
            labels = labels[:-remainder]
        
        total_plots = 2 * self.outputs
        columns = 2 # depth & sld
        rows = total_plots // columns # If 2-layer system: total_plots=2*2=4, rows=4//2=2
        position = range(1, total_plots+1)

        column_headers = ["Depth", "SLD"]
        row_headers = ["Layer {}".format(row+1) for row in range(rows)]
        pad = 5
        fig = plt.figure(figsize=(15,10))
        for k in range(total_plots):
            ax = fig.add_subplot(rows, columns, position[k])
            ax.scatter(labels[:,k], self.preds[:,k], alpha=0.2)
            
            if k == 0:
                ax.set_title(column_headers[k])
            elif k == 1:
                ax.set_title(column_headers[k])
            
            if k % 2 == 0:
                ax.set_xlabel("Ground truth: depth")
                ax.set_ylabel("Prediction: depth")
                ax.set_xlim(-100, 3000)
                ax.set_ylim(-100, 3000)
                ax.annotate(row_headers[k//2], xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                            xycoords=ax.yaxis.label, textcoords="offset points",
                            size="large", ha="right", va="center")
            else:
                ax.set_xlabel("Ground truth: SLD")
                ax.set_ylabel("Prediction: SLD")
                ax.set_xlim(-0.1, 1.1)
                ax.set_ylim(-0.1, 1.1)
        
        plt.savefig(savepath)

    def save(self, savepath):
        try:
            os.makedirs(savepath)
            print('Created path: ' + savepath)
        except OSError:
            pass

        with open(os.path.join(savepath, 'history.json'), 'w') as f:
            json_dump = convert_to_float(self.history.history)
            json_dump['timetaken'] = self.time_taken
            json.dump(json_dump, f)

        model_yaml = self.model.to_yaml()

        with open(os.path.join(savepath, 'model.yaml'), 'w') as yaml_file:
            yaml_file.write(model_yaml)

        self.model.save_weights(os.path.join(savepath, 'model_weights.h5'))

        with open(os.path.join(savepath, 'summary.txt'), 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))

        self.model.save(os.path.join(savepath, 'full_model.h5'))

class DataLoader(Sequence):
    ''' Use Keras sequence to load image data from h5 file '''
    def __init__(self, h5_file, dim, channels, batch_size, layers):
        'Initialisation'
        self.file       = h5_file                 # H5 file to read
        self.dim        = dim                     # Image dimensions
        self.channels   = channels                # Image channels                   
        self.batch_size = batch_size              # Batch size
        self.layers     = layers
        self.on_epoch_end()

    def __len__(self):
        'Denotes number of batches per epoch'
        return int(np.floor(len(np.array(self.file['images'])[100:200]) / self.batch_size))

    def __getitem__(self, index):
        'Generates one batch of data'
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        images, targets = self.__data_generation(indexes)

        return images, targets

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'
        images = np.empty((self.batch_size, *self.dim, self.channels))
        targets_depth = np.empty((self.batch_size, self.layers), dtype=float)
        targets_sld = np.empty((self.batch_size, self.layers), dtype=float)

        for i, idx in enumerate(indexes):
            image = self.file['images'][idx]
            values = self.file['Y'][idx]

            length = len(values)
            difference = length - self.layers * 2

            if difference:
                values = values[:-difference]

            images[i,] = image
            targets_depth[i,] = values[::2]
            targets_sld[i,] = values[1::2]

        for image, depth, sld in zip(x, y_depth, y_sld):
            # print(depth, sld)
            # # fig = plt.figure(figsize=(3,3))
            # plt.imshow(image.squeeze(), interpolation="nearest", cmap='Greys_r')
            # # plt.xlim(0,0.3)
            # # plt.ylim(1e-08,1.5) #this hadnt been set previously!
            # plt.axis("off")
            # plt.show()
            # plt.close()
        
        return images, {'depth': targets_depth, 'sld': targets_sld}

    def on_epoch_end(self):
        'Updates indexes after each epoch'    
        self.indexes = np.arange(len(self.file['images']))[100:200]

    def close_file(self):
        self.file.close()
    
def iter_sequence_infinite(sequence):
    while True:
        for item in sequence:
            yield item

def plot(preds, labels, savepath, batch_size, error):
    labels_1_a = labels[0][:,0]
    preds_1_a = preds[0][:,0]
    error_1_a = error[0][:,0]

    labels_1_b = labels[0][:,1]
    preds_1_b = preds[0][:,1]
    error_1_b = error[0][:,1]

    labels_2_a = labels[1][:,0]
    preds_2_a = preds[1][:,0]
    error_2_a = error[1][:,0]

    labels_2_b = labels[1][:,1]
    preds_2_b = preds[1][:,1]
    error_2_b = error[1][:,1]

    labels_2_c = labels[1][:,2]
    preds_2_c = preds[1][:,2]
    error_2_c = error[1][:,2]

    labels_2_d = labels[1][:,3]
    preds_2_d = preds[1][:,3]
    error_2_d = error[1][:,3]

    remainder = len(labels) % batch_size

    if remainder:
        labels = labels[:-remainder]
    
    total_plots = 6
    columns = 2 # depth & sld
    rows = total_plots // columns # If 2-layer system: total_plots=2*2=4, rows=4//2=2

    outer_grid = gridspec.GridSpec(3, 2, hspace=0.00, wspace=0.390, left=0.19, right=0.9, top=0.950, bottom=0.110)

    fig = plt.figure()
    pad = 55
    v_pad = 80

    ax0 = fig.add_subplot(outer_grid[0, 0])
    ax0.errorbar(labels_1_a, preds_1_a, error_1_a, fmt="o",mec="k",mew=.5,alpha=.6,capsize=3,color="b",zorder=-130,markersize=4)
    ax0.plot([0,1], [0,1], 'k', transform=ax0.transAxes)
    ax0.set_xlim([-250, 3250])
    ax0.set_ylim([-250, 3250])
    ax0.set_yticks([0, 1000, 2000, 3000])
    ax0.set_yticklabels([0, 1000, 2000, 3000])
    ax0.set_xticklabels([])
    # ax0.set_ylabel("$\mathregular{Depth_{predict}\ (Å)}$", fontsize=8)
    ax0.annotate("$\mathregular{1^{i}}$", xy=(0., 0.5), xytext=(-ax0.yaxis.labelpad - pad, v_pad),
                        xycoords="axes points", textcoords="offset points",
                        size="large", ha="right", va="center")

    ax0.set_facecolor("xkcd:very light blue")

    ax0_1 = fig.add_subplot(outer_grid[0, 1])
    ax0_1.errorbar(labels_1_b, preds_1_b, error_1_b,fmt="o",mec="k",mew=.5,alpha=.6,capsize=3,color="g",zorder=-130,markersize=4)
    ax0_1.plot([0,1], [0,1], 'k', transform=ax0_1.transAxes)
    ax0_1.set_xlim([-0.1, 1.1])
    ax0_1.set_ylim([-0.1, 1.1])
    ax0_1.set_yticks([0, 0.5, 1])
    ax0_1.set_yticklabels([0, 0.5, 1])
    ax0_1.set_xticklabels([])
    # ax0_1.set_ylabel("$\mathregular{SLD_{predict}\ (fm\ Å^{-3})}$", fontsize=8)
    ax0_1.set_facecolor("xkcd:very light blue")

    ax1 = fig.add_subplot(outer_grid[1, 0])
    ax1.errorbar(labels_2_a, preds_2_a, error_2_a,fmt="o",mec="k",mew=.5,alpha=.6,capsize=3,color="b",zorder=-130,markersize=4)
    ax1.plot([0,1], [0,1], 'k', transform=ax1.transAxes)
    ax1.set_xlim([-250, 3250])
    ax1.set_ylim([-250, 3250])
    ax1.set_yticks([0, 1000, 2000, 3000])
    ax1.set_yticklabels([0, 1000, 2000, 3000])
    ax1.set_xticklabels([])
    # ax1.set_xticks([])
    ax1.set_ylabel("$\mathregular{Depth_{predict}\ (Å)}$", fontsize=11, weight="bold")
    ax1.annotate("$\mathregular{2^{i}}$", xy=(0, 0.5), xytext=(-ax0.yaxis.labelpad - pad, v_pad),
                        xycoords="axes points", textcoords="offset points",
                        size="large", ha="right", va="center")
    # ax1.set_facecolor("xkcd:peach")

    ax2 = fig.add_subplot(outer_grid[1, 1])
    ax2.errorbar(labels_2_b, preds_2_b, error_2_b,fmt="o",mec="k",mew=.5,alpha=.6,capsize=3,color="g",zorder=-130,markersize=4)
    ax2.plot([0,1], [0,1], 'k', transform=ax2.transAxes)
    ax2.set_xlim([-0.1, 1.1])
    ax2.set_ylim([-0.1, 1.1])
    ax2.set_yticks([0, 0.5, 1])
    ax2.set_yticklabels([0, 0.5, 1])
    ax2.set_xticklabels([])
    # ax2.set_xticks([])
    ax2.set_ylabel("$\mathregular{SLD_{predict}\ (fm\ Å^{-3})}$", fontsize=11, weight="bold")
    # ax2.get_yaxis().set_label_coords(-0.15, -0.15)
    # ax2.set_facecolor("xkcd:peach")

    ax3 = fig.add_subplot(outer_grid[2, 0])
    ax3.errorbar(labels_2_c, preds_2_c, error_2_c,fmt="o",mec="k",mew=.5,alpha=.6,capsize=3,color="b",zorder=-130,markersize=4)
    ax3.plot([0,1], [0,1], 'k', transform=ax3.transAxes)
    ax3.set_xlim([-250, 3250])
    ax3.set_ylim([-250, 3250])
    ax3.set_yticks([0, 1000, 2000, 3000])
    ax3.set_yticklabels([0, 1000, 2000, 3000])
    ax3.set_xlabel("$\mathregular{Depth_{true}\ (Å)}$", fontsize=10, weight="bold")
    # ax3.set_ylabel("$\mathregular{Depth_{predict}\ (Å)}$", fontsize=8)
    ax3.annotate("$\mathregular{2^{ii}}$", xy=(0, 0.5), xytext=(-ax3.yaxis.labelpad - pad, v_pad),
                        xycoords="axes points", textcoords="offset points",
                        size="large", ha="right", va="center")
    # ax3.set_facecolor("xkcd:peach")

    ax4 = fig.add_subplot(outer_grid[2, 1])
    ax4.errorbar(labels_2_d, preds_2_d, error_2_d,fmt="o",mec="k",mew=.5,alpha=.6,capsize=3,color="g",zorder=-130,markersize=4)
    ax4.plot([0,1], [0,1], 'k', transform=ax4.transAxes)
    ax4.set_xlim([-0.1, 1.1])
    ax4.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax4.set_xticklabels([0, 0.25, 0.5, 0.75, 1])
    ax4.set_ylim([-0.1, 1.1])
    ax4.set_yticks([0, 0.5, 1])
    ax4.set_yticklabels([0, 0.5, 1])
    ax4.set_xlabel("$\mathregular{SLD_{true}\ (fm\ Å^{-3})}$", fontsize=10, weight="bold")
    # ax4.set_ylabel("$\mathregular{SLD_{predict}\ (fm\ Å^{-3})}$", fontsize=8)
    # ax4.set_facecolor("xkcd:peach")

    

    # fig = plt.gcf()
    # gs.tight_layout(fig)
    # plt.show()
    plt.savefig('onetwolayer.png', dpi=600)


    ### MATPLOTLIB SUBPLOTS
    # column_headers = ["Depth", "SLD"]
    # row_headers = ["Layer {}".format(row+1) for row in range(rows)]
    # pad = 5
    # fig = plt.figure(tight_layout=True)

    # for k in range(total_plots):
    #     ax = fig.add_subplot(rows, columns, position[k])
    #     ax.set_aspect(aspect='equal')
    #     if k % 2 == 0:
    #         ax.errorbar(labels[:,k], preds[:,k], error[:,k], fmt='o', mec='k', mew=.5, alpha=0.6, capsize=3, color='b', zorder=-130, markersize=3, ) 
    #     else:
    #         ax.errorbar(labels[:,k], preds[:,k], error[:,k], fmt='o', mec='k', mew=.5, alpha=0.6, capsize=3, color='g', zorder=-130, markersize=3) 
        
    #     ax.plot([0,1], [0,1], 'k', transform=ax.transAxes) # plot y=x
         
    #     if k == 0:
    #         ax.set_title(column_headers[k])
    #     elif k == 1:
    #         ax.set_title(column_headers[k])
        
    #     if k % 2 == 0:
    #         if k == total_plots - 2:
    #             ax.set_xlabel("$\mathregular{Depth_{true}\ (Å)}$")
    #         ax.set_ylabel("$\mathregular{Depth_{predict}\ (Å)}$")
    #         ax.set_xlim(-100, 3000)
    #         ax.set_ylim(-100, 3000)
    #         ax.annotate(row_headers[k//2], xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
    #                     xycoords=ax.yaxis.label, textcoords="offset points",
    #                     size="large", ha="right", va="center", rotation=90)
    #     else:
    #         if k == total_plots - 1:
    #             ax.set_xlabel("$\mathregular{SLD_{true}\ (fm\ Å^{-3})}$")
    #         ax.set_ylabel("$\mathregular{SLD_{predict}\ (fm\ Å^{-3})}$")
    #         ax.set_xlim(-0.1, 1.1)
    #         ax.set_ylim(-0.1, 1.1)

    # plt.show()
    # plt.savefig(savepath)

def main(args):
    if args.test:
        path_1_layer = r"C:/Users/mtk57988/stfc/neutron-net/neutron-net/models/investigate/regressor-1-layer-[2020-05-05T153216]/full_model.h5"
        path_2_layer = r"C:/Users/mtk57988/stfc/neutron-net/neutron-net/models/investigate/regressor-2-layer-[2020-05-06T231932]/full_model.h5"
        
        name = os.path.basename(os.path.dirname(os.path.abspath(args.test)))
        savepath = os.path.join(args.save, name)

        # model = load_model(args.test)

        model_1_layer = load_model(path_1_layer)
        model_2_layer = load_model(path_2_layer)
        
        testdir_1_layer = os.path.join(args.data, "1", "test.h5")
        testh5_1_layer = h5py.File(testdir_1_layer, "r")
        test_labels_1_layer = np.array(testh5_1_layer["Y"])

        testdir_2_layer = os.path.join(args.data, "2", "test.h5")
        testh5_2_layer = h5py.File(testdir_2_layer, "r")
        test_labels_2_layer = np.array(testh5_2_layer["Y"])

        test_loader_1_layer = DataLoader(testh5_1_layer, DIMS, CHANNELS, args.batch_size, 1)
        test_loader_2_layer = DataLoader(testh5_2_layer, DIMS, CHANNELS, args.batch_size, 2)

        scaler = pickle.load(open(os.path.join(args.data, "output_scaler.p"), "rb"))

        if args.bayesian:
            kdp_1_layer = KerasDropoutPredicter(model_1_layer, test_loader_1_layer)
            kdp_2_layer = KerasDropoutPredicter(model_2_layer, test_loader_2_layer)

            preds_1_layer = kdp_1_layer.predict(test_loader_1_layer, n_iter=1)
            preds_2_layer = kdp_2_layer.predict(test_loader_2_layer, n_iter=1)

            print("length", len(preds_1_layer[0][0]))
            sys.exit()

            ####################################################################################
            # y_predictions[n] : n = 0:mean prediction, 1:standard deviation, 2:ground truth
            # y_predictions[n][j] : j = 0:depth, 1:sld
            # y_predictions[n][j][k] : k = sample number
            # y_predictions[n][j][k][l] : l = 0:layer1, 1:layer2
            ###################################################################################
            
            depth_1, sld_1 = preds_1_layer[0][0], preds_1_layer[0][1]
            depth_2, sld_2 = preds_2_layer[0][0], preds_2_layer[0][1]

            depth_ground_1, sld_ground_1 = preds_1_layer[2][0], preds_1_layer[2][1]
            depth_ground_2, sld_ground_2 = preds_2_layer[2][0], preds_2_layer[2][1]

            depth_std_1, sld_std_1 = preds_1_layer[1][0], preds_1_layer[1][1]
            depth_std_2, sld_std_2 = preds_2_layer[1][0], preds_2_layer[1][1]

            padded_preds_1 = np.c_[depth_1[:,0], sld_1[:,0], np.zeros(len(depth_1)), np.zeros(len(sld_1))]
            padded_error_1 = np.c_[depth_std_1[:,0], sld_std_1[:,0], np.zeros(len(depth_1)), np.zeros(len(sld_1))]
            labels_1 = np.c_[depth_ground_1[:,0], sld_ground_1[:,0], np.zeros(len(depth_1)), np.zeros(len(sld_1))]
            error_1 = scaler.inverse_transform(padded_error_1)
            preds_1 = scaler.inverse_transform(padded_preds_1)

            padded_preds_2 = np.c_[depth_2[:,0], sld_2[:,0], depth_2[:,1], sld_2[:,1]]
            padded_error_2 = np.c_[depth_std_2[:,0], sld_std_2[:,0], depth_std_2[:,1], sld_std_2[:,1]]
            labels_2 = np.c_[depth_ground_2[:,0], sld_ground_2[:,0], depth_ground_2[:,1], sld_ground_2[:,1]]
            error_2 = scaler.inverse_transform(padded_error_2)
            preds_2 = scaler.inverse_transform(padded_preds_2)

            preds = [preds_1, preds_2]
            labels = [labels_1, labels_2]
            error = [error_1, error_2]

            # if args.layers == 2:
            #     padded_preds = np.c_[depth[:,0], sld[:,0], depth[:,1], sld[:,1]]
            #     labels = np.c_[depth_ground[:,0], sld_ground[:,0], depth_ground[:,1], sld_ground[:,1]]
            #     padded_error = np.c_[depth_std[:,0], sld_std[:,0], depth_std[:,1], sld_std[:,1]]
            #     error = scaler.inverse_transform(padded_error)
            #     preds = scaler.inverse_transform(padded_preds)

            # elif args.layers == 1:
            #     padded_preds = np.c_[depth[:,0], sld[:,0], np.zeros(len(depth)), np.zeros(len(sld))]
            #     padded_error = np.c_[depth_std[:,0], sld_std[:,0], np.zeros(len(depth)), np.zeros(len(sld))]
            #     labels = np.c_[depth_ground[:,0], sld_ground[:,0], np.zeros(len(depth)), np.zeros(len(sld))]
            #     error = scaler.inverse_transform(padded_error)
            #     preds = scaler.inverse_transform(padded_preds)

            plot(preds, labels, savepath, args.batch_size, error)


        else:
            preds = model.predict(test_loader, use_multiprocessing=False, verbose=1)
            depth, sld = preds[0], preds[1]
            if args.layers == 2:
                padded_preds = np.c_[depth[:,0], sld[:,0], depth[:,1], sld[:,1]]
                preds = scaler.inverse_transform(padded_preds)
            elif args.layers ==1:
                padded_preds = np.c_[depth[:,0], sld[:,0], np.zeros(len(depth)), np.zeros(len(sld))]
                preds = scaler.inverse_transform(padded_preds)

            random_sample = random.sample(range(0, len(preds)), 1000)
            sample_preds = preds[random_sample]
            sample_labels = test_labels[random_sample]

            plot(sample_preds, sample_labels, savepath, args.batch_size, args.layers, )
        # model.plot(test_labels, savepath)

    else:
        name = "regressor-%s-layer-[" % str(args.layers) + datetime.now().strftime("%Y-%m-%dT%H%M%S") + "]"
        savepath = os.path.join(args.save, name)

        # Log to CometML: need to add own api_key and details
        if args.log:
            experiment = Experiment(api_key="Qeixq3cxlTfTRSfJ2hyPlMWjk",
                                    project_name="general", workspace="xandrovich")

        traindir = os.path.join(args.data, str(args.layers), 'train.h5')
        valdir = os.path.join(args.data, str(args.layers), 'valid.h5')
        testdir = os.path.join(args.data, str(args.layers), 'test.h5')

        trainh5 = h5py.File(traindir, 'r')
        valh5 = h5py.File(valdir, 'r')
        testh5 = h5py.File(testdir, 'r')

        test_labels = np.array(testh5["Y"])

        train_loader = DataLoader(trainh5, DIMS, CHANNELS, args.batch_size, args.layers)
        valid_loader = DataLoader(valh5, DIMS, CHANNELS, args.batch_size, args.layers)
        test_loader = DataLoader(testh5, DIMS, CHANNELS, args.batch_size, args.layers)

        model = Net(DIMS, CHANNELS, args.epochs, args.dropout_rate, 
            args.learning_rate, args.workers, args.layers, args.batch_size)
        
        if args.summary:
            model.summary()

        model.train(train_loader, valid_loader)
        model.test(test_loader, args.data) 
        model.plot(test_labels, savepath)
        model.save(savepath)

def parse():
    parser = argparse.ArgumentParser(description="Keras Regressor Training")
    # Meta Parameters
    parser.add_argument("data", metavar="PATH", help="path to data directory")
    parser.add_argument("save", metavar="PATH", help="path to save directory")
    parser.add_argument("layers", metavar="N", type=int, help="no. layers of system")
    parser.add_argument("-l", "--log", action="store_true", help="boolean: log metrics to CometML?")
    parser.add_argument("-s", "--summary", action="store_true", help="show model summary")
    parser.add_argument("--test", metavar="PATH", help="path to regression model you wish to test")
    parser.add_argument("--bayesian", action="store_true", help="boolean: be bayesian?")

    # Model parameters
    parser.add_argument("-e", "--epochs", default=2, type=int, metavar="N", help="number of epochs")
    parser.add_argument("-b", "--batch_size", default=20, type=int, metavar="N", help="no. samples per batch (def:20)")
    parser.add_argument("-j", "--workers", default=1, type=int, metavar="N", help="no. data loading workers (def:1)")

    # Learning parameters
    parser.add_argument("-lr", "--learning_rate", default=0.0004, type=float, metavar="R", help="Nadam learning rate")
    parser.add_argument("-dr", "--dropout_rate", default=0.1, type=float, metavar="R", help="dropout rate" )
    return parser.parse_args()

def convert_to_float(dictionary):
	""" For saving model output to json"""
	jsoned_dict = {}
	for key in dictionary.keys():
		if type(dictionary[key]) == list:
			jsoned_dict[key] = [float(i) for i in dictionary[key]]
		else:
			jsoned_dict[key] = float(dictionary[key])
	return jsoned_dict

if __name__ == "__main__":
    args = parse()
    main(args)