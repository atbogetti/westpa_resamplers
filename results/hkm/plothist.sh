#!/bin/bash

plothist evolution pdist.h5 -o hkm_hist.pdf --range '0,16.89' --postprocess-function postprocess.adjust_plot && cp hkm_hist.pdf ~/copy
