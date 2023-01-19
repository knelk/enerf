import numpy as np
import os
import argparse
import cv2
import tqdm
import json
import shutil
import h5py
import glob
from utils.event_utils import *

from sklearn import linear_model
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Raw to png images in dir")
    parser.add_argument("--indir", help="Input raw dir.", default="EXPDIR/validation/raw")
    parser.add_argument("--start_from", help="Start idx", default=0, type=int)

    args = parser.parse_args()
    indir = args.indir
    outdir = os.path.join(os.path.dirname(indir), "raw_pngs")
    outdirc = os.path.join(outdir, "contrast_spread")
    assert "raw" in indir

    rawfiles_names = sorted(glob.glob(os.path.join(indir, "*.npy")))[args.start_from:]
    if not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)
    os.makedirs(outdirc, exist_ok=True)

    for i, raw_name in enumerate(rawfiles_names):
        raw_fname = os.path.split(raw_name)[1]
        raw_np = np.load(raw_name) * 255.
        raw_np = np.clip(raw_np, a_min=0, a_max=255)
        raw_np = np.rint(raw_np).astype(np.uint8)
        # raw_np = raw_np[:, :, [2, 1, 0]]
        cv2.imwrite(os.path.join(outdir, raw_fname[:-4] + ".png"), raw_np)

        img = (raw_np - np.min(raw_np)) / (np.max(raw_np) - np.min(raw_np)) * 255
        cv2.imwrite(os.path.join(outdirc, raw_fname[:-4] + "_spread.png"), img)


if __name__ == "__main__":
    main()
