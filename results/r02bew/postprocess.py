import numpy
import h5py
import matplotlib.pyplot as plt

def adjust_plot(hist, midpoint, binbounds):
    plt.xlim(2.6,25)
    plt.xlabel("Na$^{+}$ Cl$^{-}$ distance ($\AA$)")
