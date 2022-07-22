import logging
import math
import operator
import random
import pandas as pd

import numpy as np

import westpa
from westpa.core.we_driver import WEDriver
from westpa.core.segment import Segment
from westpa.core.states import InitialState

log = logging.getLogger(__name__)

class CustomDriver(WEDriver):
   
    def _segment_index_converter(self, mode, pcoords, curr_pcoords, curr_weights, data, target):

        if mode == "split":
            to_split_list = np.argsort(-data,axis=0)
            to_split_idx = []

            # don't split out of bounds walkers
            for idx in to_split_list:
                if data[idx] != 0 and curr_pcoords[:,-1][idx] > target:
                    to_split_idx.append(idx)
                                                          
            to_split_idx = to_split_idx[:5]

            curr_pcoords_to_split = curr_pcoords[:,-1][to_split_idx]

            mydf_split = pd.DataFrame()
            mydf_split['split index'] = to_split_idx
            mydf_split['pcoords'] = curr_pcoords_to_split
            mydf_split['weights'] = curr_weights[to_split_idx]
            mydf_split['scaled progress'] = data[to_split_idx]

            print("\n", mydf_split)

            if curr_pcoords_to_split.shape[0] > 1:
                converted_idx = np.zeros(curr_pcoords_to_split.shape[0], dtype=int)
                for idx, val in enumerate(curr_pcoords_to_split):
                    converted_idx[idx] = int(np.where(pcoords[:,0] == val)[0])
            else: 
                converted_idx = np.where(pcoords[:,0] == curr_pcoords_to_split)[0]

            return converted_idx

        if mode == "merge":
            to_merge_list = np.argsort(data,axis=0)
            to_merge_idx = []

            # don't merge out of bounds walkers
            for idx in to_merge_list:
                if data[idx] != 0 and curr_pcoords[:,-1][idx] > target:
                    to_merge_idx.append(idx)

            to_merge_idx = to_merge_idx[:6]
                    
            curr_pcoords_to_merge = curr_pcoords[:,-1][to_merge_idx]

            mydf_merge = pd.DataFrame()
            mydf_merge['merge index'] = to_merge_idx
            mydf_merge['pcoords'] = curr_pcoords_to_merge
            mydf_merge['weights'] = curr_weights[to_merge_idx]
            mydf_merge['scaled progress'] = data[to_merge_idx]

            print("\n", mydf_merge)

            if curr_pcoords_to_merge.shape[0] > 1:
                converted_idx = np.zeros(curr_pcoords_to_merge.shape[0], dtype=int)
                for idx, val in enumerate(curr_pcoords_to_merge):
                    converted_idx[idx] = int(np.where(pcoords[:,0] == val)[0])
            else: 
                converted_idx = np.where(pcoords[:,0] == curr_pcoords_to_merge)[0]

            return converted_idx


    def _split_by_data(self, bin, to_split, split_into):

        if len(to_split) > 1:
            for segment in to_split:
                bin.remove(segment)
                new_segments_list = self._split_walker(segment, split_into, bin)
                bin.update(new_segments_list)
        else:
            to_split = to_split[0]
            bin.remove(to_split)
            new_segments_list = self._split_walker(to_split, split_into, bin)
            bin.update(new_segments_list)


    def _merge_by_data(self, bin, to_merge):

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
                
                curr_pcoords = curr_pcoords.reshape(nsegs,nframes)

                # change the following for different algorithms
                start = 15.0
                target = 2.6
                
                progresses = np.zeros((nsegs), dtype=float)
                
                # find percent change between first and last frame
                for idx, ival in enumerate(curr_pcoords):
                    distance = np.abs(ival[-1]-target)/np.abs(start-target)
                    if distance > 1:
                        progress = 0
                    else:
                        progress = float(1-distance)
                    progresses[idx] = progress

                # this will prioritize trajectories with higher weights
                scaled_progresses = progresses * (1/log_weights)

                # this is for if no weights are used
                #scaled_progresses = progresses * 1

                pd.set_option('display.colheader_justify', 'center')
                mydf = pd.DataFrame()
                mydf['pcoords'] = curr_pcoords[:,-1]
                mydf['weights'] = curr_weights
                mydf['adj weights'] = 1/log_weights
                mydf['progress'] = progresses
                mydf['scaled progress'] = scaled_progresses
                print("\n", mydf.sort_values(by=['scaled progress']))

                # check if not initializing, then split and merge
                init_check = curr_pcoords[:,0] != curr_pcoords[:,-1]
                #print(init_check)

                if np.any(init_check):

                    # split walker with largest scaled diff
                    split_into = 2
                    to_split_index = self._segment_index_converter("split", pcoords, curr_pcoords, curr_weights, scaled_progresses, target)
                    to_split = np.array([segments[to_split_index]])[0]

                    self._split_by_data(bin, to_split, split_into)
    
                    # merge walker with lowest scaled diff into next lowest
                    to_merge_index = self._segment_index_converter("merge", pcoords, curr_pcoords, curr_weights, scaled_progresses, target)
                    to_merge = segments[to_merge_index]

                    self._merge_by_data(bin, to_merge)

        # another sanity check
        self._check_post()

        self.new_weights = self.new_weights or []

        log.debug('used initial states: {!r}'.format(self.used_initial_states))
        log.debug('available initial states: {!r}'.format(self.avail_initial_states))
