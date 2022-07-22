#!/bin/bash

plothist evolution pdist.h5 -o r02b_hist.pdf --range '0,16.89' --postprocess-function postprocess.adjust_plot && cp r02b_hist.pdf ~/copy
