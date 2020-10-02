import os, h5py
import numpy as np
import numpy.random
import matplotlib.pyplot as plt
from refnx.reflect import SLD, MaterialSLD, ReflectModel

class CurveGenerator:
    """The CurveGenerator class contains all code relating to reflectivity curve generation with refnx.

    Class Attributes:
        roughness (int): the default roughness between each layer in Angstrom.

    """
    roughness = 2
    qMin = 0.005
    scale = 1
    dq  = 2
    points = 200
    bkg_rate = 5e-7
    noise_constant = 100
    thick_bounds = (20,3000)

    @staticmethod
    def bias_thickness(thick_bounds):
        thick_range = np.arange(*thick_bounds, 10)

        # The following biases choices towards thinner layers
        thick_probs = []
        for i in range(len(thick_range)):
            thick_probs.append(1.0 / (thick_range[i] + 100))
        thick_probs /= sum(thick_probs)
        
        return thick_range, thick_probs

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
    def sample_noise(q, r, file="./data/directbeam_noise.dat", constant=100):
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
    qMax = 0.3 
    bkg = 0
    substrate_sld = 2.047
    sld_bounds = (-1,10)
    
    @staticmethod
    def generate(generate_num, layers):
        """Generates `generate_num` curves with given number of layers, bounds on SLD and thickness,
           and substrate SLD. Bias is placed on thickness towards thinner layers.

        Args:
            generate_num (int): the number of curves to generate.
            layers (int): the number of layers for each curve to be generated with.

        Returns:
            A list of `generate_num` refnx Structure objects.

        """
        
        #Discretise SLD and thickness ranges.
        sld_range = np.arange(*NeutronGenerator.sld_bounds, 0.1)
        thick_range, thick_probs = CurveGenerator.bias_thickness(CurveGenerator.thick_bounds)
        return [NeutronGenerator.__random_structure(layers, sld_range, thick_range, thick_probs) for i in range(generate_num)]

    @staticmethod
    def __random_structure(layers, sld_range, thick_range, thick_probs):
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
            component = NeutronGenerator.__make_component(sld_range, thick_range, thick_probs, substrate=False)
            structure = structure | component
        substrate = NeutronGenerator.__make_component(sld_range, thick_range, thick_probs, substrate=True)
        structure = structure | substrate
        return structure

    @staticmethod
    def __make_component(sld_range, thick_range, thick_probs, substrate=False):
        """Generates a single refnx component object representing a layer of the structure.

        Args:
            sld_range (ndarray): the discrete range of valid SLD values for generation.
            thick_range (ndarray): the range of valid thickness (depth) values for generation.
            thick_probs (list): the probabilities for each discrete thickness value.
            substrate (Boolean): whether the component is the substrate or not.

        Returns:
            refnx Component object.

        """
        if substrate:
            thickness = 0 #Substrate has 0 thickness.
            sld = NeutronGenerator.substrate_sld
        else:
            thickness = np.random.choice(thick_range, p=thick_probs)
            sld = np.random.choice(sld_range)

        return SLD(sld)(thick=thickness, rough=CurveGenerator.roughness)
    
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

        with h5py.File(save_path + "/{}-Layer.h5".format(name), 'w') as file:
            parameters = []
            data = []
            q = np.linspace(CurveGenerator.qMin, NeutronGenerator.qMax, CurveGenerator.points) #Use range of q values specified.

            for i, structure in enumerate(structures):
                model = ReflectModel(structure, bkg=NeutronGenerator.bkg, 
                                     scale=CurveGenerator.scale, dq=CurveGenerator.dq)
                r = model(q) #Generate r values.

                if noisy: #Add background and sample noise if specified.
                    r_noisy = CurveGenerator.background_noise(r, bkg_rate=CurveGenerator.bkg_rate)
                    r = CurveGenerator.sample_noise(q, r_noisy, constant=CurveGenerator.noise_constant)

                data.append(list(zip(q, r))) #Add (q, r) pairs as a list to the data to store.

                temp = [0, 0, 0, 0, 0, 0] #Designed for parameter for up to 3 layers.
                for i, component in enumerate(structure.components[1:-1]): #Exclude air and substrate
                    temp[2*i]   = component.thick.value
                    temp[2*i+1] = component.sld.real.value
                parameters.append(temp)
                
            num_curves = len(structures)
            file.create_dataset("SLD_NUMS", data=parameters, chunks=(num_curves, 6))
            file.create_dataset("DATA",     data=data,       chunks=(num_curves, CurveGenerator.points, 2))


class XRayGenerator(CurveGenerator):
    wavelength = 1.54
    material   = 'H20'
    density_constant = 9.4691e-6
    qMax = 1
    bkg = 1e-9
    density_bounds = (0.5, 16)
    Si_density = 2.1 #Density to set water at to get SLD of Si

    @staticmethod
    def generate(generate_num, layers):
        #Discretise SLD and thickness ranges.
        density_range = np.arange(*XRayGenerator.density_bounds, 0.1)
        thick_range, thick_probs = CurveGenerator.bias_thickness(CurveGenerator.thick_bounds)
        return [XRayGenerator.__random_structure(layers, density_range, thick_range, thick_probs) for i in range(generate_num)]

    @staticmethod
    def __random_structure(layers, density_range, thick_range, thick_probs):
        #The structure consists of air followed by each layer and then finally the substrate.
        structure = SLD(0, name="Air")
        for i in range(layers):
            component = XRayGenerator.__make_component(density_range, thick_range, thick_probs, substrate=False)
            structure = structure | component
        substrate = XRayGenerator.__make_component(density_range, thick_range, thick_probs, substrate=True)
        structure = structure | substrate
        return structure

    @staticmethod
    def __make_component(density_range, thick_range, thick_probs, substrate):
        if substrate:
            thickness = 0 #Substrate has 0 thickness.
            density = XRayGenerator.Si_density
        else:
            thickness = np.random.choice(thick_range, p=thick_probs)
            density = np.random.choice(density_range)
            
        SLD = MaterialSLD(XRayGenerator.material, density, probe='x-ray', wavelength=XRayGenerator.wavelength)
        return SLD(thick=thickness, rough=CurveGenerator.roughness)
    
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

        with h5py.File(save_path + "/{}-Layer.h5".format(name), 'w') as file:
            parameters = []
            data = []
            q = np.linspace(CurveGenerator.qMin, XRayGenerator.qMax, CurveGenerator.points) #Use range of q values specified.

            for i, structure in enumerate(structures):
                model = ReflectModel(structure, bkg=XRayGenerator.bkg, 
                                     scale=CurveGenerator.scale, dq=CurveGenerator.dq)
                r = model(q) #Generate r values.

                if noisy: #Add background and sample noise if specified.
                    r_noisy = CurveGenerator.background_noise(r, bkg_rate=CurveGenerator.bkg_rate)
                    r = CurveGenerator.sample_noise(q, r_noisy, constant=CurveGenerator.noise_constant)

                data.append(list(zip(q, r))) #Add (q, r) pairs as a list to the data to store.

                temp = [0, 0, 0, 0, 0, 0] #Designed for parameter for up to 3 layers.
                for i, component in enumerate(structure.components[1:-1]): #Exclude air and substrate
                    temp[2*i]   = component.thick.value
                    temp[2*i+1] = component.sld.density.value * XRayGenerator.density_constant #Convert density to SLD
                parameters.append(temp)

            num_curves = len(structures)
            file.create_dataset("SLD_NUMS", data=parameters, chunks=(num_curves, 6))
            file.create_dataset("DATA",     data=data,       chunks=(num_curves, CurveGenerator.points, 2))


if __name__ == "__main__":
    save_path  = './models/investigate/data'
    layers     = ['one', 'two', 'three']
    num_curves = 5000
    xray       = False
    noisy      = False

    for i, name in enumerate(layers):
        layers = i+1
        print(">>> Generating {}-layer curves".format(layers))
        if xray:
            structures = XRayGenerator.generate(num_curves, layers)
            XRayGenerator.save(save_path, name, structures, noisy=noisy)
        else:
            structures = NeutronGenerator.generate(num_curves, layers)
            NeutronGenerator.save(save_path, name, structures, noisy=noisy)
