import logging
import math
import operator
import random

import numpy as np

import westpa
from westpa.core.we_driver import WEDriver
from westpa.core.segment import Segment
from westpa.core.states import InitialState

log = logging.getLogger(__name__)

class DummyDriver(WEDriver):
   
    def _segment_index_converter(self, mode, pcoords, curr_pcoords, scaled_diffs):

        if mode == "split":
            to_split_idx = np.argmin(scaled_diffs)
            curr_pcoords_to_split = curr_pcoords[:,-1][to_split_idx]
            converted_idx = int(np.where(pcoords[:,0] == curr_pcoords_to_split)[0])

            return converted_idx

        if mode == "merge":
            to_merge_idx = np.argsort(-scaled_diffs,axis=0)[:2]
            curr_pcoords_to_merge = curr_pcoords[:,-1][to_merge_idx]
            if curr_pcoords_to_merge.shape[0] > 1:
                converted_idx = np.zeros(curr_pcoords_to_merge.shape[0], dtype=int)
                for idx, val in enumerate(curr_pcoords_to_merge):
                    converted_idx[idx] = int(np.where(pcoords[:,0] == val)[0])
            else: 
                converted_idx = np.where(pcoords[:,0] == curr_pcoords_to_merge)[0]

            return converted_idx


    def _split_by_diff(self, bin, to_split, split_into):

        bin.remove(to_split)
        new_segments_list = self._split_walker(to_split, split_into, bin)
        bin.update(new_segments_list)


    def _merge_by_diff(self, bin, to_merge, cumul_weight):

        bin.difference_update(to_merge)
        new_segment, parent = self._merge_walkers(to_merge, None, bin)
        bin.add(new_segment)


    def _run_we(self):
        '''Run recycle/split/merge. Do not call this function directly; instead, use
        populate_initial(), rebin_current(), or construct_next().'''
        self._recycle_walkers()

        # sanity check
        self._check_pre()

        # dummy resampling block
        for bin in self.next_iter_binning:
            if len(bin) == 0:
                continue
            else:
                # this will just get you the final pcoord for each segment... which may not be enough
                segments = np.array(sorted(bin, key=operator.attrgetter('weight')), dtype=np.object_)
                pcoords = np.array(list(map(operator.attrgetter('pcoord'), segments)))
                weights = np.array(list(map(operator.attrgetter('weight'), segments)))

                log_weights = -1 * np.log(weights)
 
                nsegs = pcoords.shape[0]
                nframes = pcoords.shape[1]

                pcoords = pcoords.reshape(nsegs,nframes)

                # this will allow you to get the pcoords for all frames
                current_iter_segments = self.current_iter_segments

                curr_segments = np.array(sorted(current_iter_segments, key=operator.attrgetter('weight')), dtype=np.object_)
                curr_pcoords = np.array(list(map(operator.attrgetter('pcoord'), curr_segments)))
                curr_weights = np.array(list(map(operator.attrgetter('weight'), curr_segments)))

                log_weights = -1 * np.log(weights)
 
                nsegs = pcoords.shape[0]
                nframes = pcoords.shape[1]

                diffs = np.zeros((nsegs))

                curr_pcoords = curr_pcoords.reshape(nsegs,nframes)
                
                # find percent change between first and last frame
                for idx, ival in enumerate(curr_pcoords):
                    diff = ((ival[-1] - ival[0]) / ival[0]) * 100
                    diffs[idx] = diff

                diffs[diffs > 0] = 0

                init_check = np.any(diffs)

                scaled_diffs = diffs * log_weights
                
                # print for sanity check
                #print("pcoords", pcoords[:,0])
                #print("current pcoords", curr_pcoords[:,-1])
                #print("weights", weights)
                #print("log_weights", log_weights)
                #print("diffs", diffs)
                #print("scaled_diffs", scaled_diffs)

                if init_check:

                    # split walker with largest scaled diff
                    split_into = 2
                    to_split_index = self._segment_index_converter("split", pcoords, curr_pcoords, scaled_diffs)
                    to_split = segments[to_split_index]

                    self._split_by_diff(bin, to_split, split_into)
    
                    # merge walker with lowest scaled diff into next lowest
                    cumul_weight = np.add.accumulate(weights)
                    to_merge_index = self._segment_index_converter("merge", pcoords, curr_pcoords, scaled_diffs)
                    to_merge = segments[to_merge_index]

                    self._merge_by_diff(bin, to_merge, cumul_weight)

        # another sanity check
        self._check_post()

        self.new_weights = self.new_weights or []

        log.debug('used initial states: {!r}'.format(self.used_initial_states))
        log.debug('available initial states: {!r}'.format(self.avail_initial_states))
