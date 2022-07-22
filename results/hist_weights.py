import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys

h5filepath = sys.argv[1]

h5file = h5py.File(h5filepath+"/west.h5")

weight_list = []

for i in range(1,101):
    weights = h5file["iterations/iter_" + str(i).zfill(8) + "/seg_index"]['weight']
    for idx, val in enumerate(weights):
        weight_list.append(val)

weight_list = np.array(weight_list)

logbins = np.geomspace(1e-20, 1, 50)
plt.hist(weight_list, bins=logbins)
plt.xscale('log')
plt.xlim(1e-20,1)
plt.xlabel("trajectory weight")
plt.ylabel("counts")
plt.savefig("weighthist_%s.pdf"%h5filepath)

plt.clf()
weight_list = np.loadtxt(h5filepath+"/succ_weights.txt")

logbins = np.geomspace(1e-20, 1, 50)
plt.hist(weight_list, bins=logbins, color="orange")
plt.xscale('log')
plt.xlim(1e-20,1)
plt.xlabel("trajectory weight")
plt.ylabel("counts")
plt.savefig("succweighthist_%s.pdf"%h5filepath)
