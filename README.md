# neutron-net
![neutron-curve](/resources/real_sim.png) <br />
A lightweight tool to analyse neutron reflectivity curves and generate initial refnx fits.

## About The Project
A neural network-based tool for automatic estimation of thin film thicknesses and scattering length densities from neutron reflectivity curves of systems containing up to three layers on top of a substrate. Raw instrument data (in our case from the OFFSPEC Neutron Reflectometer) is processed and passed through the neural networks to produce layer parameter predictions. These predictions are then fed into refnx, and serve as initial "guesses" for further optimisation to yield a model that likely describes a given sample. All that is required from the end user is running the `pipeline.py` file with their reflectivity data.

The project was motivated by the desire to enable on-experiment analysis of reflectivity data, informing choices about changing experiment conditions or samples <em>in operando</em>.

### Built With
* TensorFlow Keras
* Python
* refnx

## Getting Started
### Environment and Installation
To replicate development environment with the Anaconda distribution, create an empty conda environment, and run: <br />
```conda install --file requirements.txt```

### Usage
The system consists of 6 files. To operate the system as a whole, only `pipeline.py` needs to be run. If you wish to use the other files independently of one another this is also possible; please read the individual sections for each component if this is the case.

#### The Pipeline
If you have not trained any models, you can set up the pipeline by calling `Pipeline.setup` which will, by default, generate reflectivity data, convert to images, train a classifier and train regressors using the data. You will need to provide the layers for which you wish to set the pipeline up for, e.g. [1,2,3] for up to 3-layer structures. You can also modify the number of curves generated (which will significantly impact runtime), chunk size for h5 storage, and the number of epochs to train the classifier and regressors for. If you have already generated data and converted it to images, you can set the generate_data flag to false to train on your own data. Likewise, if you already have trained a classifier, this can be loaded by setting the train_classifier flag to false. Finally, if you have already trained the regressors for each layer, you can load these instead of training from scratch by setting the train_regressor flag to false.

To run the pipeline, you can call the `Pipeline.run` method. You must provide a path to a directory containing .dat files to predict on (these should be CSV files of the form X, Y, Error). You will also need to provide the file paths of a trained classifier as well as a dictionary of file paths for trained regressors for each layer you wish to classify. By running the pipeline, the provided classifier will be used for a layer prediction. This will then be used to determine the regressor to use when predicting each layer's SLD and thickness using dropout prediction. These predictions are then fed into a refnx model and plotted against the given data. Fitting can optionally be applied with these predictions as initial estimates.

#### Synthetic Data Generation
Data can be generated with refnx using the `generate_refnx.py` file. Specifically, `CurveGenerator.generate` allows for generation of <em>n</em> refnx Structure objects with a specified number of layers, SLDs and thicknesses within given bounds. The code is currently setup to use a silicon substrate with 2.047 Å SLD. A default 2 Å roughness between layers is included to simulate real data. Thicknesses choices are biased towards tinner layers. 

The `CurveGenerator.save` method can then be used to store these `n` structures as reflectivity curves in h5 format. The min, max and number of momentum transfer values can be specified along with background, scale and resolution parameters (defaults are 0, 1 and 2 respectively). The option to add sample and background noise is also available with the noisy flag. This requires the included directbeam_noise.dat sample.

#### Creating Images
Images are required as input to the classifier and regressors and `generate_data.py` facilitates creation of these images. Data is split into training, validation and test splits for use in these models.

The `generate_images` function can be called with the file path to a directory containing h5 files (in the correct format) to generate images for. Also provided to this function is a list of the layers for which images are to be generated for the corresponding file. For example, passing [1,2,3] will create images for any files containing 'one', 'two' or 'three' in their file name in the given data path. These will then be saved together in the given save path directory. This allows for files with curves of a specific layer to be created, as is required for regression, as well as for files containing curves of multiple different layers, as is required for classification. 

Depending on the number of curves being generated, the chunk size for the h5 files can be modified for potential speedup. Please note that this process can take some time with large numbers of curves and will potentially generate large files (~6GB for 50,000 curves). During creation of these images of the input reflectivity curves, targets are scaled to be between 0 and 1 for speeding up training and the data is shuffled.

#### Merging Files
To merge train.h5, validate.h5 and test.h5 files, you can call the `merge` function in `merge_data.py`. After creating separate h5 files for each layer for each of the regressors, these files can be merged together to get a combined file for use in training the classifier. This allows for reuse of data in the classifier as well as the regressors and also allows for images to be generated on separate machines before being combined.

#### Classification
To perform classification, call the `classify` function in `classification.py`. You will need to provide a file path to the data (train.h5, validate.h5 and test.h5) to test and/or train on. Optionally, a save path can be provided to save a newly trained model to. A load path could also be provided to load an existing classifier for further training and evaluation; the train flag indicates whether to perform training or not. Hyperparameters that can be adjusted include the number of epochs to train for, the learning rate, batch size and dropout rate. The classifier will work with up to 3-layer classification.

#### Regression
To perform regression, call the `regress` function in `regression.py`. You will need to provide a file path to the data containing data for a specific layer (train.h5, validate.h5 and test.h5) to test and/or train on. You must also provide the layer for which the regressor is being used for. Optionally, a save path can be provided to save a newly trained model to. A load path could also be provided to load an existing regressor for further training and evaluation; the train flag indicates whether to perform training or not. Hyperparameters that can be adjusted include the number of epochs to train for, the learning rate, batch size and dropout rate. 

## Contributing
Contributions are gladly accepted and would be very much appreciated.

  1. Fork the project
  2. Create your feature branch (```git checkout -b feature/AmazingFeature```)
  3. Commit your changes (```git commit -m 'Add some AmazingFeature'```)
  4. Push to the branch (```git push origin feature/AmazingFeature```)
  5. Open a pull request

## License
Distributed under the GNU AGPLv3 license. See ```LICENSE``` for more information.

## Contact
Daniil Mironov - daniil.mironov@stfc.ac.uk\
Jos Cooper     - jos.cooper@stfc.ac.uk\
James Durant   - james.durant@stfc.ac.uk

## Acknowledgements
Many thanks for the collaboration of the Scientific Machine Learning group at the Rutherford Appleton Laboratory, their expertise and assistance with this project - in particular, we'd like to thank Rebecca Mackenzie for being so key in laying the groundwork, and giving the project a strong foundation. The authors would like to thank Johnathan Xue, whose initial work and results allowed the project to go ahead. We thank the Ada Lovelace Centre for funding.
