# neutron-net
![neutron-curve](/neutron-net/resources/real_sim.png) <br />
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
```conda install --file requirements_dev.txt```

### Usage
The system consists of 6 files. To operate the system as a whole, only `pipeline.py` needs to be run.

#### Synthetic Data Generation
Data can be generated with refnx using the `generate_refnx.py` file. Specifically, `CurveGenerator.generate` allows for generation of ``n`` refnx Structure objects with a specified number of layers, SLDs and thicknesses within given bounds. The code is currently setup to use a silicon substrate with 2.047 Å SLD. A default 2 Å roughness between layers is included to simulate real data. Thicknesses choices are biased towards tinner layers. 

The `CurveGenerator.save` method can then be used to store these `n` structures as reflectivity curves in h5 format. The min, max and number of momentum transfer values can be specified along with background, scale and resolution parameters (defaults are 0, 1 and 2 respectively). The option to add sample and background noise is also available with the noisy flag. This requires the included directbeam_noise.dat sample.

#### Creating Images
Images are required as input to the classifier and regressors and `generate_data.py` facilitates creation of these images. Data is split into training, validation and test splits for use in these models.

The `generate_images` function can be called with the file path to a directory containing h5 files (in the correct format) to generate images for. Also provided to this function is a list of the layers for which images are to be generated for the corresponding file. For example, passing [1,2,3] will create images for any files containing 'one', 'two' or 'three' in their file name in the given data path. These will then be saved together in the given save path directory. This allows for files with curves of a specific layer to be created, as is required for regression, as well as for files containing curves of multiple different layers, as is required for classification. 

Depending on the number of curves being generated, the chunk size for the h5 files can be modified for potential speedup. Please note that this process can take some time with large numbers of curves and will potentially generate large files (~6GB for 50,000 curves). During creation of these images of the input reflectivity curves, targets are scaled to be between 0 and 1 for speeding up training and the data is shuffled.

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
Jos Cooper     - jos.cooper@stfc.ac.uk

## Acknowledgements
Many thanks for the collaboration of the Scientific Machine Learning group at the Rutherford Appleton Laboratory, their expertise and assistance with this project - in particular, we'd like to thank Rebecca Mackenzie for being so key in laying the groundwork, and giving the project a strong foundation. The authors would like to thank Johnathan Xue, whose initial work and results allowed the project to go ahead. We thank the Ada Lovelace Centre for funding.
