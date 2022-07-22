#!/bin/bash

plothist evolution pdist.h5 -o hkf_hist.pdf --range '0,16.89' --postprocess-function postprocess.adjust_plot && cp hkf_hist.pdf ~/copy
