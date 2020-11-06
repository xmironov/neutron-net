import os, h5py
import numpy as np
import numpy.random
import matplotlib.pyplot as plt
from refnx.reflect import SLD, MaterialSLD, ReflectModel

class CurveGenerator:
    """The CurveGenerator class contains all code relating to general reflectivity curve generation with refnx.

    Class Attributes:
        rough_bounds (tuple): bounds on the roughness between each layer in Angstrom.
        qMin (float): the minimum Q value to use when generating data.
        scale (int): the value for the scale instrument parameter.
        dq (int): the instrument resolution parameter.
        points (int): the number of data points to generate for each sample.
        bkg_rate (float): the background rate used for adding background noise.
        noise_constant (int): a value to control the level of sample noise applied.
        thick_bounds (tuple): the range of values layer thicknesses can take.

    """
    rough_bounds    = (2,8)
    qMin            = 0.005
    qMax            = 0.3
    scale           = 1
    dq              = 2
    points          = 500
    bkg_rate        = 5e-7
    noise_constant  = 100
    thick_bounds    = (20,1000)

    @staticmethod
    def bias_thickness(thick_bounds):
        """Biases a given thickness range towards its lower end.

        Args:
            thick_bounds (tuple): the range of thicknesses to bias.

        Returns:
            A discretised range along with the probability of choosing each thickness.

        """
        thick_range = np.arange(*thick_bounds, 10) #Discretise range.

        thick_probs = []
        for i in range(len(thick_range)):
            thick_probs.append(1.0 / (thick_range[i] + 100)) #Biases choices towards lower values
        thick_probs /= sum(thick_probs)

        return thick_range, thick_probs

    @staticmethod
    def plot_SLD(structure):
        """Plots the SLD profile for a given refnx Structure object.

        Args:
            structure (Structure): the structure to plot the SLD profile for.

        """
        plt.figure(1, dpi=600)
        plt.plot(*structure.sld_profile())
        plt.ylabel('SLD /$10^{-6} \AA^{-2}$')
        plt.xlabel('distance / $\AA$')
        plt.show()

    @staticmethod
    def plot_reflectivity(q, r):
        """Plots the reflectivity profile for a given Structure object.

        Args:
            q (ndarray): the range of momentum transfer values to plot (x-axis).
            r (ndarray): the range of reflectance values to plot (y-axis).

        """
        plt.figure(dpi=600)
        plt.plot(q, r, c="black")
        plt.xlabel("$\mathregular{Q\ (Å^{-1})}$")
        plt.ylabel('Reflectivity (arb.)')
        plt.yscale('log')
        plt.show()

    @staticmethod
    def sample_noise(q, r, file="../resources/directbeam_noise.dat", constant=100):
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
    def background_noise(r, bkg_rate=5e-7):
        """Applies background noise to given reflectance values.

        Args:
            r (ndarray): the range of reflectance values.
            bkg_rate (type): the background rate value.

        """
        #Background signal always ADDs to the signal.
        #Sometimes background could be 0. In which case it does not contribute to the signal
        return [r_point + max(np.random.normal(1, 0.5) * bkg_rate, 0) for r_point in r]


class NeutronGenerator(CurveGenerator):
    """The NeutronGenerator class contains all code relating to neutron reflectivity curve generation.

    Class Attributes:
        qMax (int): the maximum Q value to use when generating data.
        bkg (float): the value of the background to apply when generating.
        substrate_sld (float): the scattering length density of the substrate.
        sld_bounds (tuple): the range of values layer SLDs can take.

    """
    qMax          = 0.3
    bkg           = 0
    substrate_sld = 2.047
    sld_bounds    = (-1,10)

    @staticmethod
    def generate(generate_num, layers):
        """Generates `generate_num` curves with given number of layers.

        Args:
            generate_num (int): the number of curves to generate.
            layers (int): the number of layers for each curve to be generated with.

        Returns:
            A list of `generate_num` refnx Structure objects.

        """
        sld_range = np.arange(*NeutronGenerator.sld_bounds, 0.1) #Discretise SLD range.
        thick_range, thick_probs = CurveGenerator.bias_thickness(CurveGenerator.thick_bounds) #Apply thickness bias.
        return [NeutronGenerator.__random_structure(layers, sld_range, thick_range, thick_probs) for i in range(generate_num)]

    @staticmethod
    def __random_structure(layers, sld_range, thick_range, thick_probs):
        """Generates a single random refnx Structure object with desired parameters.

        Args:
            layers (int): the number of layers for each curve to be generated with.
            sld_range (ndarray): the discrete range of valid SLD values for generation.
            thick_range (ndarray): the range of valid thickness (depth) values for generation.
            thick_probs (list): the probabilities for each discrete thickness value.

        Returns:
            A refnx Structure object.

        """
        #The structure consists of air followed by each layer and then finally the substrate.
        roughness = np.random.choice(np.arange(*CurveGenerator.rough_bounds, 0.5)) #Select a random roughness for the layers in this sample.
        structure = SLD(0, name="Air")
        for i in range(layers):
            component = NeutronGenerator.__make_component(sld_range, thick_range, thick_probs, roughness, substrate=False)
            structure = structure | component
        substrate = NeutronGenerator.__make_component(sld_range, thick_range, thick_probs, roughness, substrate=True)
        structure = structure | substrate
        return structure

    @staticmethod
    def __make_component(sld_range, thick_range, thick_probs, roughness, substrate=False):
        """Generates a single refnx component object representing a layer of the structure.

        Args:
            sld_range (ndarray): the discrete range of valid SLD values for generation.
            thick_range (ndarray): the range of valid thickness (depth) values for generation.
            thick_probs (list): the probabilities for each discrete thickness value.
            roughness (float): the given roughness for the layer.
            substrate (Boolean): whether the component is the substrate or not.

        Returns:
            A refnx Component object.

        """
        if substrate:
            thickness = 0 #Substrate has 0 thickness in refnx.
            sld       = NeutronGenerator.substrate_sld
        else:
            thickness = np.random.choice(thick_range, p=thick_probs) #Select a random thickness and SLD.
            sld       = np.random.choice(sld_range)

        return SLD(sld)(thick=thickness, rough=roughness)

    @staticmethod
    def save(save_path, name, structures, noisy=False):
        """Saves a list of Structure objects in the HDF5 format.

        Args:
            save_path (string): the file path to save the HDF5 file to.
            name (string): the name of the HDF5 file.
            structures (list): a list of refnx Structure objects to save.
            noisy (Boolean): whether to add background and sample noise when saving.

        """
        save_path = save_path + "/" + name
        if not os.path.exists(save_path): #Create directories if not present.
            os.makedirs(save_path)

        #Use space q points in equal log bins.
        q = np.logspace(np.log10(CurveGenerator.qMin), np.log10(CurveGenerator.qMax), CurveGenerator.points)

        parameters = []
        data = []
        for structure in structures:
            model = ReflectModel(structure, bkg=NeutronGenerator.bkg,
                                 scale=CurveGenerator.scale, dq=CurveGenerator.dq)

            r = model(q) #Generate r values.

            if noisy: #Add background and sample noise if specified.
                r_noisy = CurveGenerator.background_noise(r, bkg_rate=CurveGenerator.bkg_rate)
                r = CurveGenerator.sample_noise(q, r_noisy, constant=CurveGenerator.noise_constant)

            data.append(list(zip(q, r))) #Add (q, r) pairs as a list to the data to store.

            #CurveGenerator.plot_reflectivity(q, r)

            temp = [0, 0, 0, 0, 0, 0] #Designed for parameter for up to 3 layers.
            for i, component in enumerate(structure.components[1:-1]): #Exclude air and substrate
                temp[2*i]   = component.thick.value
                temp[2*i+1] = component.sld.real.value
            parameters.append(temp)

        with h5py.File(save_path + "/{}-Layer.h5".format(name), 'w') as file:
            file.create_dataset("SLD_NUMS", data=parameters, chunks=(len(structures), 6))
            file.create_dataset("DATA",     data=data,       chunks=(len(structures), CurveGenerator.points, 2))


class XRayGenerator(CurveGenerator):
    """The XRayGenerator class contains all code relating to x-ray reflectivity curve generation.

    Class Attributes:
        qMax (int): the maximum Q value to use when generating data.
        bkg (float): the value of the background to apply when generating.
        substrate_density (float): the mass density of the substrate in g / cm**3
        density_constant (float): constant used for converting densities to SLDs.
        density_bounds (tuple): the range of mass densities each layer can take.
        wavelength (float): wavelength of radiation (Angstrom)
        material (string): chemical formula of the substrate.

    """
    bkg               = 1e-9
    substrate_density = 2.1 #The density to set water at in order to get the SLD of Si
    density_constant  = 9.4691
    density_bounds    = (0.5, 16)
    wavelength        = 1.54
    material          = 'H2O' #H2O is just used to generate a range of SLD values

    @staticmethod
    def generate(generate_num, layers):
        """Generates `generate_num` curves with given number of layers.

        Args:
            generate_num (int): the number of curves to generate.
            layers (int): the number of layers for each curve to be generated with.

        Returns:
            A list of `generate_num` refnx Structure objects.

        """
        density_range = np.arange(*XRayGenerator.density_bounds, 0.1) #Discretise density range.
        thick_range, thick_probs = CurveGenerator.bias_thickness(CurveGenerator.thick_bounds) #Bias thicknesses.
        return [XRayGenerator.__random_structure(layers, density_range, thick_range, thick_probs) for i in range(generate_num)]

    @staticmethod
    def __random_structure(layers, density_range, thick_range, thick_probs):
        """Generates a single random refnx Structure object with desired parameters.

        Args:
            layers (int): the number of layers for each curve to be generated with.
            density_range (ndarray): the discrete range of valid density values for generation.
            thick_range (ndarray): the range of valid thickness (depth) values for generation.
            thick_probs (list): the probabilities for each discrete thickness value.

        Returns:
            A refnx Structure object.

        """
        #The structure consists of air followed by each layer and then finally the substrate.
        roughness = np.random.choice(np.arange(*CurveGenerator.rough_bounds, 0.5)) #Select a random roughness for the layers in this sample.
        structure = SLD(0, name="Air")
        for i in range(layers):
            component = XRayGenerator.__make_component(density_range, thick_range, thick_probs, roughness, substrate=False)
            structure = structure | component
        substrate = XRayGenerator.__make_component(density_range, thick_range, thick_probs, roughness, substrate=True)
        structure = structure | substrate
        return structure

    @staticmethod
    def __make_component(density_range, thick_range, thick_probs, roughness, substrate=False):
        """Generates a single refnx component object representing a layer of the structure.

        Args:
            density_range (ndarray): the discrete range of valid density values for generation.
            thick_range (ndarray): the range of valid thickness (depth) values for generation.
            thick_probs (list): the probabilities for each discrete thickness value.
            roughness (float): the given roughness for the component.
            substrate (Boolean): whether the component is the substrate or not.

        Returns:
            A refnx Component object.

        """
        if substrate:
            thickness = 0 #Substrate has 0 thickness in refnx.
            density = XRayGenerator.substrate_density
        else:
            thickness = np.random.choice(thick_range, p=thick_probs)
            density = np.random.choice(density_range)

        SLD = MaterialSLD(XRayGenerator.material, density, probe='x-ray', wavelength=XRayGenerator.wavelength)
        return SLD(thick=thickness, rough=roughness)

    @staticmethod
    def save(save_path, name, structures, noisy=False):
        """Saves a list of Structure objects in the HDF5 format.

        Args:
            save_path (string): the file path to save the HDF5 file to.
            name (string): the name of the HDF5 file.
            structures (list): a list of refnx Structure objects to save.
            noisy (Boolean): whether to add background and sample noise when saving.

        """
        save_path = save_path + "/" + name
        if not os.path.exists(save_path): #Create directories if not present.
            os.makedirs(save_path)

        #Space q points in equal log bins.
        q = np.logspace(np.log10(CurveGenerator.qMin), np.log10(CurveGenerator.qMax), CurveGenerator.points)

        parameters = []
        data = []
        for structure in structures:
            model = ReflectModel(structure, bkg=XRayGenerator.bkg,
                                 scale=CurveGenerator.scale, dq=CurveGenerator.dq)
            r = model(q) #Generate r values.

            if noisy: #Add background and sample noise if specified.
                r_noisy = CurveGenerator.background_noise(r, bkg_rate=CurveGenerator.bkg_rate)
                r = CurveGenerator.sample_noise(q, r_noisy, constant=CurveGenerator.noise_constant)

            data.append(list(zip(q, r))) #Add (q, r) pairs as a list to the data to store.

            #CurveGenerator.plot_reflectivity(q, r)

            temp = [0, 0, 0, 0, 0, 0] #Designed for parameter for up to 3 layers.
            for i, component in enumerate(structure.components[1:-1]): #Exclude air and substrate
                temp[2*i]   = component.thick.value
                temp[2*i+1] = component.sld.density.value * XRayGenerator.density_constant #Convert density to SLD
            parameters.append(temp)

        with h5py.File(save_path + "/{}-Layer.h5".format(name), 'w') as file:
            file.create_dataset("SLD_NUMS", data=parameters, chunks=(len(structures), 6))
            file.create_dataset("DATA",     data=data,       chunks=(len(structures), CurveGenerator.points, 2))


if __name__ == "__main__":
    save_path  = './models/neutron/data'
    layers     = ['one', 'two', 'three']
    num_curves = 50000
    xray       = False
    noisy      = True

    for i, name in enumerate(layers):
        layers = i+1
        print(">>> Generating {}-layer curves".format(layers))
        if xray:
            structures = XRayGenerator.generate(num_curves, layers)
            XRayGenerator.save(save_path, name, structures, noisy=noisy)
        else:
            structures = NeutronGenerator.generate(num_curves, layers)
            NeutronGenerator.save(save_path, name, structures, noisy=noisy)
