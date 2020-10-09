# neutron-net
![neutron-curve](/resources/real_sim.png) <br/>
A lightweight tool to analyse neutron reflectivity curves and generate initial refnx fits.

## About The Project
A neural network-based tool for automatic estimation of thin film thicknesses and scattering length densities from neutron reflectivity curves of systems containing up to three layers on top of a substrate. Raw instrument data (in our case from the OFFSPEC Neutron Reflectometer) is processed and passed through the neural networks to produce layer parameter predictions. These predictions are then fed into refnx, and serve as initial "guesses" for further optimisation to yield a model that likely describes a given sample. All that is required from the end user is running the [pipeline.py](/neutron-net/pipeline.py) file with their reflectivity data.

The project was motivated by the desire to enable on-experiment analysis of reflectivity data, informing choices about changing experiment conditions or samples <em>in operando</em>.

### Built With
* TensorFlow Keras
* Python
* refnx

## Getting Started
### Installation
1. To replicate development environment with the [`Anaconda`](https://www.anaconda.com/products/individual) distribution, first create an empty conda environment by running: <br /> ```conda create --name neutron-net```

2. To activate the environment, run: ```conda activate neutron-net```.

3. Install pip by running: ```conda install pip```

4. Navigate to the main neutron-net directory. Run the following to install the required packages from the [requirements.txt](/requirements.txt) file: <br />
   ```pip install -r requirements.txt```

5. You should be able to run the code. Try: ```python pipeline.py```

### Usage
Please read [usage](/resources/usage.md) for instructions on how to use the system.

## Contributing
Contributions are gladly accepted and would be very much appreciated.

  1. Fork the project
  2. Create your feature branch (```git checkout -b feature/AmazingFeature```)
  3. Commit your changes (```git commit -m 'Add some AmazingFeature'```)
  4. Push to the branch (```git push origin feature/AmazingFeature```)
  5. Open a pull request

## License
Distributed under the GNU AGPLv3 license. See [license](/LICENSE) for more information.

## Contact
Daniil Mironov - daniil.mironov@stfc.ac.uk\
Jos Cooper     - jos.cooper@stfc.ac.uk\
James Durant   - james.durant@stfc.ac.uk

## Acknowledgements
Many thanks for the collaboration of the Scientific Machine Learning group at the Rutherford Appleton Laboratory, their expertise and assistance with this project - in particular, we'd like to thank Rebecca Mackenzie for being so key in laying the groundwork, and giving the project a strong foundation. The authors would like to thank Johnathan Xue, whose initial work and results allowed the project to go ahead. We thank the Ada Lovelace Centre for funding.
