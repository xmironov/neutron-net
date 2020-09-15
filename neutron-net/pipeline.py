import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Suppress TensorFlow warnings

from generate_refnx import CurveGenerator
from generate_data  import generate_images
from merge_data     import merge
from classification import classify
from regression     import regress

LAYERS_STR = {1: "one", 2: "two", 3: "three"}

def setup(save_path, layers=[1,2,3], curve_num=5000, chunk_size=1000, generate_data=True, train_classifier=True, train_regressor=True):
    save_path = './models/investigate/test'
    layers = [1, 2]
    curve_num  = 500
    chunk_size = 50 
    generate_data = False
    train_classifier = False
    train_regressor  = False
    
    if generate_data:
        print("-------------- Generating Data ------------")
        for layer in layers:
            print(">>> Generating {}-layer curves".format(layer))
            structures = CurveGenerator.generate(curve_num, layer, substrate_SLD=2.047)
            CurveGenerator.save(save_path + "/data", LAYERS_STR[layer], structures)
            
            print(">>> Creating images for {}-layer curves".format(layer))
            save_path_layer = data_path_layer = save_path + "/data/{}".format(LAYERS_STR[layer])
            generate_images(data_path_layer, save_path_layer, [layer], chunk_size=chunk_size, display_status=False)
        
        layers_paths = [save_path + "/data/{}".format(LAYERS_STR[layer]) for layer in layers]
        merge(save_path + "/data", layers_paths)
    
    print("\n-------------- Classification -------------")
    if train_classifier:
        print(">>> Training classifier")
        classify(save_path + "/data/merged", save_path, train=True, epochs=1)
    else:
        print(">>> Loading classifier")
        load_path = save_path + "/classifier/full_model.h5"
        classify(save_path + "/data/merged", load_path=load_path, train=False)
    
    print("\n---------------- Regression ---------------")
    for layer in layers:
        data_path_layer = save_path + "/data/{}".format(LAYERS_STR[layer])
        if train_regressor:
            print(">>> Training {}-layer regressor".format(LAYERS_STR[layer]))
            regress(data_path_layer, layer, save_path, epochs=1)
        else:
            print(">>> Loading {}-layer regressor".format(LAYERS_STR[layer]))
            load_path_layer = save_path + "/{}-layer-regressor/full_model.h5".format(LAYERS_STR[layer])
            regress(data_path_layer, layer, load_path=load_path_layer, train=False)
        print()

if __name__ == "__main__":
    save_path = './models/investigate/test'
    layers = [1, 2]
    curve_num  = 500
    chunk_size = 50 
    generate_data = False
    train_classifier = False
    train_regressor  = False
    
    setup(save_path, layers, curve_num, chunk_size, generate_data, train_classifier, train_regressor)