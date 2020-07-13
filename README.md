# neutron-net
![neutron-curve](/neutron-net/resources/real_sim.png) <br />
A lightweight tool to analyse neutron reflectivity curves and generate initial GenX fits.

## About The Project
A neural network based tool for automatic estimation of thin film thicknesses and scattering length densities from neutron reflectivity curves of systems containing up to two layers on top of a substrate. Raw instrument data (in our case from the OFFSPEC Neutron Reflectometer) is processed and passed through the neural networks to produce layer parameter predictions. These predictions are then fed into GenX, and serve as initial "guesses" for further optimisation to yield a model that likely describes a given sample. All that is required from the end user is opening the fitting software and pressing the "fit", and "simulate" buttons.

The project was motivated by the desire to enable on-experiment analysis of reflectivity data, informing choices about changing experiment conditions or samples <em>in operando</em>.

### Built With
* TensorFlow Keras
* Python
* GenX

## Getting Started
### Environment
To replicate development environment with the Anaconda distribution, create an empty conda environment, and run: <br />
```conda install --file requirements_dev.txt -c comet_ml```
### Installation


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
