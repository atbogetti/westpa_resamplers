import h5py
import numpy
import sys

h5filepath = sys.argv[1]

nsegs = numpy.sum(h5py.File(h5filepath+"/west.h5", "r")['summary']['n_particles'])
agg_simtime = nsegs*2/1000
print("aggregate simulation time (ns):", agg_simtime)
