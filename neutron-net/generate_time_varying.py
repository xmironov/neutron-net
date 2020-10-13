import os.path
import numpy as np
from refnx.reflect import SLD, ReflectModel
from generate_refnx import CurveGenerator, NeutronGenerator

def generate_time_varying(save_path):
    if not os.path.exists(save_path): #Create directories if not present.
        os.makedirs(save_path)
    
    sld_range = np.arange(*NeutronGenerator.sld_bounds, 0.1) #Discretise SLD range.
    thick_range = np.arange(100, 850, 50) 
    q = np.linspace(CurveGenerator.qMin, NeutronGenerator.qMax, CurveGenerator.points) #Use range of q values specified.
        
    for thickness in thick_range:
        #The structure consists of air followed by each layer and then finally the substrate.
        air       = SLD(0, name="Air")
        layer1    = SLD(np.random.choice(sld_range), name="Layer 1")(thick=100, rough=CurveGenerator.roughness)
        layer2    = SLD(np.random.choice(sld_range), name="Layer 2")(thick=thickness, rough=CurveGenerator.roughness)
        substrate = SLD(NeutronGenerator.substrate_sld, name="Si Substrate")(thick=0, rough=CurveGenerator.roughness)
        
        structure = air | layer1 | layer2 | substrate
        model = ReflectModel(structure, bkg=NeutronGenerator.bkg, scale=CurveGenerator.scale, dq=CurveGenerator.dq)
        r = model(q)
        
        r_noisy_bkg    = CurveGenerator.background_noise(r, bkg_rate=CurveGenerator.bkg_rate)
        r_noisy_sample = CurveGenerator.sample_noise(q, r_noisy_bkg, constant=CurveGenerator.noise_constant)
        
        data = np.zeros((CurveGenerator.points, 2))
        data[:, 0] = q
        data[:, 1] = r_noisy_sample
        np.savetxt(save_path+"/{}.csv".format(thickness), data, delimiter=",")
        
        CurveGenerator.plot_reflectivity(q, r)

        
if __name__ == "__main__":     
    save_path  = './models/two-layer-time-varying'
    generate_time_varying(save_path)