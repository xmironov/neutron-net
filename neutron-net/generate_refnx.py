import os, h5py, random
import numpy as np
import matplotlib.pyplot as plt
from refnx.reflect import SLD, ReflectModel

class CurveGenerator:
    @staticmethod
    def generate(generate_num, layers, sld_bounds=(-0.5,10), thick_bounds=(20,3000), substrate_SLD=None):
        return [CurveGenerator.__random_structure(layers, sld_bounds, thick_bounds, substrate_SLD) for i in range(generate_num)]

    @staticmethod
    def __random_structure(layers, sld_bounds, thick_bounds, substrate_SLD):
        structure = SLD(0, name="Air")
        for i in range(layers):
            component = CurveGenerator.__make_component(sld_bounds, thick_bounds, substrate=False)
            structure = structure | component
        substrate = CurveGenerator.__make_component(sld_bounds, thick_bounds, True, substrate_SLD)
        structure = structure | substrate
        return structure

    @staticmethod
    def __make_component(sld_bounds, thick_bounds, substrate=False, substrate_SLD=None):
        if substrate:
            thickness = 0
        else:
            thickness = random.uniform(thick_bounds[0], thick_bounds[1]) #Could try to bias these towards thinner thicknesses?

        if not substrate or substrate_SLD is None:
            sld = random.uniform(sld_bounds[0], sld_bounds[1])
        else:
            sld = substrate_SLD

        return SLD(sld)(thick=thickness)

    @staticmethod
    def plot_SLD(structure):
        plt.figure(1)
        plt.plot(*structure.sld_profile())
        plt.ylabel('SLD /$10^{-6} \AA^{-2}$')
        plt.xlabel('distance / $\AA$')

    @staticmethod
    def plot_reflectivity(q, r):
        plt.figure(2)
        plt.plot(q, r)
        plt.xlabel('Q')
        plt.ylabel('Reflectivity')
        plt.yscale('log')

    @staticmethod
    def save(save_path, name, structures, qMin=0.005, qMax=0.3, points=200, bkg=0, scale=1, dq=5):
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
                data.append(list(zip(q, r)))

                #CurveGenerator.plot_reflectivity(q, r)

                temp = [0, 0, 0, 0, 0, 0]
                for i, component in enumerate(structure.components[1:-1]): #Exclude air and substrate
                    temp[2*i]   = component.thick.value
                    temp[2*i+1] = component.sld.real.value
                parameters.append(temp)

            file.create_dataset("SLD_NUMS", data=parameters, chunks=True)
            file.create_dataset("DATA",     data=data,       chunks=True)

if __name__ == "__main__":
    save_path = './models/investigate/test/data'
    layers = ['one', 'two']
    
    for i, name in enumerate(layers):
        layers = i+1
        print(">>> Generating {}-layer curves".format(layers))
        structures = CurveGenerator.generate(500, layers, substrate_SLD=2.047)
        CurveGenerator.save(save_path, name, structures)
