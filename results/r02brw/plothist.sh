#!/bin/bash

plothist evolution pdist.h5 -o r02brw_hist.pdf --range '0,16.89' --postprocess-function postprocess.adjust_plot && cp r02brw_hist.pdf ~/copy
