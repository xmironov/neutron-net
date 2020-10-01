import os, h5py
import numpy as np
import numpy.random
import matplotlib.pyplot as plt
from refnx.reflect import SLD, ReflectModel

class CurveGenerator:
    """The CurveGenerator class contains all code relating to reflectivity curve generation with refnx.

    Class Attributes:
        roughness (int): the default roughness between each layer in Angstrom.

    """
    roughness = 2

    @staticmethod
    def generate(generate_num, layers, sld_bounds=(-1,10), thick_bounds=(20,3000), substrate_SLD=None):
        """Generates `generate_num` curves with given number of layers, bounds on SLD and thickness,
           and substrate SLD. Bias is placed on thickness towards thinner layers.

        Args:
            generate_num (int): the number of curves to generate.
            layers (int): the number of layers for each curve to be generated with.
            sld_bounds (tuple): the range of valid SLD values for generation.
            thick_bounds (tuple): the range of valid thickness (depth) values for generation.
            substrate_SLD (float): the SLD of the substrate. Default is None.

        Returns:
            A list of `generate_num` refnx Structure objects.

        """
        #Discretise SLD and thickness ranges.
        sld_range   = np.arange(sld_bounds[0], sld_bounds[1], 0.1)
        thick_range = np.arange(thick_bounds[0], thick_bounds[1], 10)

        # The following biases choices towards thinner layers
        thick_probs = []
        for i in range(len(thick_range)):
            thick_probs.append(1.0 / (thick_range[i] + 100))
        thick_probs /= sum(thick_probs)

        return [CurveGenerator.__random_structure(layers, sld_range, thick_range, thick_probs, substrate_SLD) for i in range(generate_num)]

    @staticmethod
    def __random_structure(layers, sld_range, thick_range, thick_probs, substrate_SLD):
        """Generates a single random refnx Structure object with desired parameters.

        Args:
            layers (int): the number of layers for each curve to be generated with.
            sld_range (ndarray): the discrete range of valid SLD values for generation.
            thick_range (ndarray): the range of valid thickness (depth) values for generation.
            thick_probs (list): the probabilities for each discrete thickness value.
            substrate_SLD (float): the SLD of the substrate.

        Returns:
            refnx Structure object.

        """
        #The structure consists of air followed by each layer and then finally the substrate.
        structure = SLD(0, name="Air")
        for i in range(layers):
            component = CurveGenerator.__make_component(sld_range, thick_range, thick_probs, substrate=False)
            structure = structure | component
        substrate = CurveGenerator.__make_component(sld_range, thick_range, thick_probs, substrate=True, substrate_SLD=substrate_SLD)
        structure = structure | substrate
        return structure

    @staticmethod
    def __make_component(sld_range, thick_range, thick_probs, substrate=False, substrate_SLD=None):
        """Generates a single refnx component object representing a layer of the structure.

        Args:
            sld_range (ndarray): the discrete range of valid SLD values for generation.
            thick_range (ndarray): the range of valid thickness (depth) values for generation.
            thick_probs (list): the probabilities for each discrete thickness value.
            substrate (Boolean): whether the component is the substrate or not.
            substrate_SLD (float): the SLD of the substrate.

        Returns:
            refnx Component object.

        """
        if substrate:
            thickness = 0 #Substrate has 0 thickness.
        else:
            thickness = np.random.choice(thick_range, p=thick_probs)

        if not substrate or substrate_SLD is None:
            sld = np.random.choice(sld_range)
        else:
            sld = substrate_SLD #Use given substrate sld.

        return SLD(sld)(thick=thickness, rough=CurveGenerator.roughness)

    @staticmethod
    def plot_SLD(structure):
        """Plots the SLD profile for a given Structure object.

        Args:
            structure (Structure): the structure to plot the SLD profile for.

        """
        plt.figure(1)
        plt.plot(*structure.sld_profile())
        plt.ylabel('SLD /$10^{-6} \AA^{-2}$')
        plt.xlabel('distance / $\AA$')

    @staticmethod
    def plot_reflectivity(q, r, figure=2):
        """Plots the reflectivity profile for a given Structure object.

        Args:
            q (ndarray): the range of momentum transfer values to plot (x-axis).
            r (ndarray): the range of reflectance values to plot (y-axis).
            figure (int): the figure identifier.

        """
        plt.figure(figure)
        plt.plot(q, r)
        plt.xlabel('Q')
        plt.ylabel('Reflectivity')
        plt.yscale('log')

    @staticmethod
    def save(save_path, name, structures, noisy=False, qMin=0.005, qMax=0.3, points=200, bkg=0, scale=1, dq=2):
        """Saves a list of Structure objects in the HDF5 format.

        Args:
            save_path (string): the file path to save the HDF5 file to.
            name (string): the name of the HDF5 file.
            structures (list): a list of refnx Structure objects to save.
            noisy (Boolean): whether to add background and sample noise when saving.
            qMin (int): the minimum q value to use when generating r values.
            qMax (int): the maximum q value to use when generating r values.
            points (int): the number of q values to use.
            bkg (float): the value of the background to add when saving.
            scale (int): the instrument scale parameter.
            dq (int): the instrument resolution parameter.

        """
        save_path = save_path + "/" + name
        if not os.path.exists(save_path): #Create directories if not present.
            os.makedirs(save_path)

        with h5py.File(save_path + "/{}-Layer.h5".format(name), 'w') as file:
            parameters = []
            data = []
            q = np.linspace(qMin, qMax, points) #Use range of q values specified.

            for i, structure in enumerate(structures):
                model = ReflectModel(structure, bkg=bkg, scale=scale, dq=dq)
                r = model(q) #Generate r values.

                if noisy: #Add background and sample noise if specified.
                    r_noisy = CurveGenerator.__background_noise(r, bkg_rate=5e-7)
                    r = CurveGenerator.__sample_noise(q, r_noisy, constant=1000)

                data.append(list(zip(q, r))) #Add (q, r) pairs as a list to the data to store.

                temp = [0, 0, 0, 0, 0, 0] #Designed for parameter for up to 3 layers.
                for i, component in enumerate(structure.components[1:-1]): #Exclude air and substrate
                    temp[2*i]   = component.thick.value
                    temp[2*i+1] = component.sld.real.value
                parameters.append(temp)

            file.create_dataset("SLD_NUMS", data=parameters, chunks=True)
            file.create_dataset("DATA",     data=data,       chunks=True)

    @staticmethod
    def __sample_noise(q, r, file="./data/directbeam_noise.dat", constant=1000):
        """Adds noise to given reflectivity data using the direct beam sample.

        Args:
            q (ndarray): the range of q (momentum transfer) values.
            r (ndarray): the range of r (reflectance) values.
            file (string): the file path to the directbeam_noise file.
            constant (int): the sample noise constant.

        Returns:
            Reflectance values with sample noise added.

        """
        try: #Try to load the beam sample file: exit the function if not found.
            direct_beam = np.loadtxt(file, delimiter=',')[:, 0:2]
        except OSError:
            print("directbeam_noise.dat file not found")
            return

        flux_density = np.interp(q, direct_beam[:, 0], direct_beam[:, 1]) #Not all Q values are the same
        r_noisy = []
        for i, r_point in zip(flux_density, r): #Beam interp against simulated reflectance.
            normal_width = r_point * constant / i
            r_noisy.append(np.random.normal(loc=r_point, scale=normal_width)) #Using beam interp
        return r_noisy

    @staticmethod
    def __background_noise(r, bkg_rate=5e-7):
        """Applies background noise to given reflectance values.

        Args:
            r (ndarray): the range of reflectance values.
            bkg_rate (type): the background rate value.

        """
        #Background signal always ADDs to the signal.
        #Sometimes background could be 0. In which case it does not contribute to the signal
        return [r_point + max(np.random.normal(1, 0.5) * bkg_rate, 0) for r_point in r]


if __name__ == "__main__":
    save_path = './models/investigate/data'
    layers = ['one', 'two', 'three']
    num_curves = 1000

    for i, name in enumerate(layers):
        layers = i+1
        print(">>> Generating {}-layer curves".format(layers))
        structures = CurveGenerator.generate(num_curves, layers, substrate_SLD=2.047)
        CurveGenerator.save(save_path, name, structures, noisy=False)
