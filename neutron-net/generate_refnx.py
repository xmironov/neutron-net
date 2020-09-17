import os, h5py, random
import numpy as np
import numpy.random
import matplotlib.pyplot as plt
from refnx.reflect import SLD, ReflectModel

class CurveGenerator:
    roughness = 2
    
    @staticmethod
    def generate(generate_num, layers, sld_bounds=(-0.5,10), thick_bounds=(20,3000), substrate_SLD=None):
        # The following biases choices towards thinner layers
        thick_range = np.arange(thick_bounds[0], thick_bounds[1], 10)
        thick_probs = []
        for i in range(len(thick_range)):
            thick_probs.append(1.0 / (thick_range[i] + 100))
        thick_probs /= sum(thick_probs)
        
        return [CurveGenerator.__random_structure(layers, sld_bounds, thick_range, thick_probs, substrate_SLD) for i in range(generate_num)]

    @staticmethod
    def __random_structure(layers, sld_bounds, thick_range, thick_probs, substrate_SLD):
        structure = SLD(0, name="Air")
        for i in range(layers):
            component = CurveGenerator.__make_component(sld_bounds, thick_range, thick_probs, substrate=False)
            structure = structure | component
        substrate = CurveGenerator.__make_component(sld_bounds, thick_range, thick_probs, True, substrate_SLD)
        structure = structure | substrate
        return structure

    @staticmethod
    def __make_component(sld_bounds, thick_range, thick_probs, substrate=False, substrate_SLD=None):
        if substrate:
            thickness = 0
        else:
            thickness = np.random.choice(thick_range, p=thick_probs)

        if not substrate or substrate_SLD is None:
            sld = random.uniform(sld_bounds[0], sld_bounds[1])
        else:
            sld = substrate_SLD

        return SLD(sld)(thick=thickness, rough=CurveGenerator.roughness)

    @staticmethod
    def plot_SLD(structure):
        plt.figure(1)
        plt.plot(*structure.sld_profile())
        plt.ylabel('SLD /$10^{-6} \AA^{-2}$')
        plt.xlabel('distance / $\AA$')

    @staticmethod
    def plot_reflectivity(q, r, figure=2):
        plt.figure(figure)
        plt.plot(q, r)
        plt.xlabel('Q')
        plt.ylabel('Reflectivity')
        plt.yscale('log')

    @staticmethod
    def save(save_path, name, structures, noisy=False, qMin=0.005, qMax=0.3, points=200, bkg=0, scale=1, dq=2):
        save_path = save_path + "/" + name
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        with h5py.File(save_path + "/{}-Layer.h5".format(name), 'w') as file:
            parameters = []
            data = []
            q = np.linspace(qMin, qMax, points)

            for i, structure in enumerate(structures):
                model = ReflectModel(structure, bkg=bkg, scale=scale, dq=dq)
                r = model(q)
                
                #CurveGenerator.plot_reflectivity(q, r, figure=2)
                if noisy:
                    r_noisy = CurveGenerator.__background_noise(r, bkg_rate=5e-7)
                    r = CurveGenerator.__sample_noise(q, r_noisy, constant=1000)
                    #CurveGenerator.plot_reflectivity(q, r, figure=3)
                
                data.append(list(zip(q, r)))

                temp = [0, 0, 0, 0, 0, 0]
                for i, component in enumerate(structure.components[1:-1]): #Exclude air and substrate
                    temp[2*i]   = component.thick.value
                    temp[2*i+1] = component.sld.real.value
                parameters.append(temp)

            file.create_dataset("SLD_NUMS", data=parameters, chunks=True)
            file.create_dataset("DATA",     data=data,       chunks=True)

    @staticmethod
    def __sample_noise(q, r, file="./data/directbeam_noise.dat", constant=1000):
        """Add noise using the direct beam sample"""
        try:
            direct_beam = np.loadtxt(file, delimiter=',')[:, 0:2]
        except OSError:
            print("directbeam_noise.dat file not found")
            return
        
        flux_density = np.interp(q, direct_beam[:, 0], direct_beam[:, 1]) #this is done because not all Q values are the same
        r_noisy = []
        for i, r_point in zip(flux_density, r): #beam interp against simulated reflectance.
            normal_width = r_point * constant / i
            r_noisy.append(np.random.normal(loc=r_point, scale=normal_width)) #using beam interp
    
        return r_noisy
    
    @staticmethod
    def __background_noise(r, bkg_rate=5e-7):
        """Apply background noise"""
        #Background signal always ADDs to the signal (r).
        #Sometimes background could be 0. In which case it does not contribute to the signal
        return [r_point + max(np.random.normal(1, 0.5) * bkg_rate, 0) for r_point in r]
    

if __name__ == "__main__":
    save_path = './models/investigate/data'
    layers = ['one', 'two']
    
    for i, name in enumerate(layers):
        layers = i+1
        print(">>> Generating {}-layer curves".format(layers))
        structures = CurveGenerator.generate(500, layers, substrate_SLD=2.047)
        CurveGenerator.save(save_path, name, structures, noisy=False)
