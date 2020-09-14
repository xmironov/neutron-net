import h5py
import matplotlib.pyplot as plt
filename = "./models/investigate/classification/test/merge/test.h5"
#filename = "./models/investigate/classification/given/train.h5"

with h5py.File(filename, "r") as f:
    print("Keys: %s" % f.keys())
    data = list(f['inputs'])
    
    for i in range(len(data)):
        q = []
        r = []
        print(list(f['layers'])[i])
        #print(list(f['targets'])[i])
        for each in data[i]:
            q.append(each[0])
            r.append(each[1])
        
        #plt.plot(q,r)
        #plt.yscale("log")