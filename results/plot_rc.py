import h5py
import matplotlib.pyplot as plt
import numpy
import sys

h5filepath = sys.argv[1]

data = h5py.File(h5filepath+"/ANALYSIS/TEST/direct.h5", "r")['rate_evolution']['expected'][:,1,0]

data /= (2e-12*0.002774)

xs = numpy.arange(0,data.shape[0])*2

plt.axhline(3.9e9, linestyle='--', linewidth=2, color="grey")
plt.fill_between(xs, 3.9e9+0.3e9, 3.9e9-0.3e9, color="grey", alpha=0.4)
plt.semilogy(xs, data, linewidth=2)
plt.xlabel("molecular time (ps)")
plt.ylabel("rate constant estimate (M$^{-1}$s$^{-1}$)")
plt.xlim(0, 198)
plt.ylim(1e5, 1e11)
plt.savefig("rc_%s.pdf"%h5filepath)
