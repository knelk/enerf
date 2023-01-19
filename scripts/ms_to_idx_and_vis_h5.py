import numpy as np
import os
import argparse
import cv2
import tqdm
import json
import shutil
import h5py
import hdf5plugin
import glob
from utils.event_utils import *

from sklearn import linear_model
import pandas as pd

import yaml
from numba import jit, njit
import math
from utils.plot_utils import render_ev_accumulation
from utils.event_utils import compute_ms_to_idx

"""
Descriptions:

This script prepares h5, i.e. it computes ms_to_idx and performs visualizations.
"""


def main():
    parser = argparse.ArgumentParser(description="Resizes images in dir")
    parser.add_argument(                               
        "--infile", help="Input ev file.", default="/DATADIR/events.h5"
    )
    parser.add_argument('--dt_ms', type=int, default=50, help="ms")
    parser.add_argument('--H', type=int, default=720, help="ms")
    parser.add_argument('--W', type=int, default=1280, help="ms")
    args = parser.parse_args()

    print(f"\n \n Compute ms_to_idx + visualize: file {args.infile}")
    assert ".h5" in args.infile

    # reading events
    h5file = os.path.join(args.infile)
    ef_in = h5py.File(h5file, "r+") 
        
    N_evs = len(ef_in["events"]['x'])
    print(f"Got {ef_in['events']['x'].shape} evs in {ef_in['events']['t'][0]*1e-6}secs to {ef_in['events']['t'][N_evs-1]*1e-6}secs")

    if "t_offset" in ef_in.keys():
        offset_us = ef_in['t_offset'][:]
    else:
        offset_us = 0
    print(f"offset is {offset_us}")

    tss_us = ef_in['events']["t"][:]
    print(f"computing ms_to_idx for {N_evs} tss")
    ms_to_idx = compute_ms_to_idx(tss_us * 1000)
    if "ms_to_idx" not in ef_in.keys():
        ef_in.create_dataset('ms_to_idx', shape=ms_to_idx.shape, dtype=np.uint64)
        ef_in['ms_to_idx'][:] = ms_to_idx
    else: 
        assert np.sum(ef_in['ms_to_idx']-ms_to_idx) == 0

    outvizfolder = os.path.join(os.path.dirname(args.infile), f"evs_vis_dt_{args.dt_ms}_ms")
    os.makedirs(outvizfolder, exist_ok=True)
    event_slicer = EventSlicer(ef_in)

    N_slices = int((ef_in['events']["t"][-1]-ef_in['events']["t"][0])/1e6*1000/args.dt_ms) # 50ms => 20Hz
    tss_slice_us = np.linspace(ef_in['events']["t"][0], ef_in['events']["t"][-1], N_slices)
    
    print(f"Visualizing (distorted) {N_slices} event slices.")
    pbar = tqdm.tqdm(total=len(tss_slice_us)-1)
    for i in range(len(tss_slice_us)-1):
        start_time_us = tss_slice_us[i]
        end_time_us = tss_slice_us[i+1]
        ev_batch = event_slicer.get_events(start_time_us, end_time_us)
        if ev_batch is None:
            continue
        p = ev_batch['p']
        x = ev_batch['x']
        y = ev_batch['y']
        img = render_ev_accumulation(x, y, p, args.H, args.W)
        fpath = os.path.join(outvizfolder,  "%06d" % i + ".png")
        cv2.imwrite(fpath, img)
        pbar.update(1)
    print("Done visualizing events \n \n")


if __name__ == "__main__":
    main()
