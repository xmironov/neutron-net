import os.path
import numpy as np
from refnx.reflect import SLD, ReflectModel
from generate_refnx import CurveGenerator, NeutronGenerator

def generate_time_varying(save_path):
    """Generates a series of datasets simulating an experiement with a layer whose
       thickness changes over time.

    Args:
        save_path (string): the file path to save the datasets to.

    """
    if not os.path.exists(save_path): #Create directories if not present.
        os.makedirs(save_path)

    thick_min   = 100
    thick_max   = 900
    thick_step  = 50
    thick_range = np.arange(thick_min, thick_max, thick_step) #Range of thicknesses over the experiement duration.
    
    points = 250
    q = np.linspace(CurveGenerator.qMin, NeutronGenerator.qMax, points) #Range of Q values.
    
    layer1_sld = 2.5
    layer2_sld = 5
    layer2_thickness = 100
    print("Layer 1 - Thickness: [{0},{1}] | SLD: {2:7.4f}".format(thick_min, thick_max, layer1_sld))
    print("Layer 2 - Thickness: {0}       | SLD: {1:7.4f}".format(layer2_thickness, layer2_sld))

    for thickness in thick_range: #Iterate over each thickness the top layer will take.
        #The structure consists of air followed by each layer and then finally the substrate.
        air       = SLD(0, name="Air")
        layer1    = SLD(layer1_sld, name="Layer 1")(thick=thickness, rough=CurveGenerator.roughness)
        layer2    = SLD(layer2_sld, name="Layer 2")(thick=layer2_thickness, rough=CurveGenerator.roughness)
        substrate = SLD(NeutronGenerator.substrate_sld, name="Si Substrate")(thick=0, rough=CurveGenerator.roughness)

        structure = air | layer1 | layer2 | substrate
        model = ReflectModel(structure, bkg=NeutronGenerator.bkg, scale=CurveGenerator.scale, dq=CurveGenerator.dq)
        r = model(q)

        #Add simulated noise to the data.
        r_noisy_bkg    = CurveGenerator.background_noise(r, bkg_rate=CurveGenerator.bkg_rate)
        r_noisy_sample = CurveGenerator.sample_noise(q, r_noisy_bkg, constant=CurveGenerator.noise_constant)

        data = np.zeros((points, 3))
        data[:, 0] = q
        data[:, 1] = r_noisy_sample
        data[:, 2] = 1e-10 #Error is set to be (near) zero as it is not used by the networks. This could be improved.
        np.savetxt(save_path+"/{}.dat".format(thickness), data, delimiter="    ")

if __name__ == "__main__":
    save_path  = './data/two-layer-time-varying'
    generate_time_varying(save_path)
