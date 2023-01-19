from genericpath import exists
from math import dist
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

def compute_psnr(img0, img1, max_val=255):
    assert max_val > 0.00000001
    assert img0.shape == img1.shape
    return -10 * np.log10(np.mean(np.power(img0.astype(np.float32) - img1.astype(np.float32), 2))) + 20 * np.log10(max_val)


def main():
    parser = argparse.ArgumentParser(description="Resizes images in dir")
    parser.add_argument(        
        "--indir", help="Input image directory.", default="DATADIR/mocap-desk2/e2vids/left/e2vid_up4_freq0/e2calib/"
    )
    args = parser.parse_args()

    print(f"Undistorting {args.indir}")
    assert "e2vid" in args.indir or "e2calib" in args.indir
    assert "TUMVIE" in args.indir
    
    W, H = 1280, 720
    K_evs = np.zeros((3,3))
    dist_coeffs = np.array([-0.11519655713574485, -0.06222183183004903, 0.21682612342850954, -0.23528623774744806])
    K_evs[0,0] = 1049.5830934616608 ##### TUMVIE mocapdesk2
    K_evs[0,2] = 634.7184038833433
    K_evs[1,1] = 1049.4229746040553
    K_evs[1,2] = 263.46974530961836
    K_evs[2, 2] = 1 
    K_new_evs = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K_evs, dist_coeffs, (W, H), np.eye(3), balance=0.5) 

    img_list = sorted(glob.glob(os.path.join(args.indir) + "*.png"))
    img_list = sorted([os.path.join(args.indir, im) for im in img_list if im.endswith(".png")])
    H, W, _ = cv2.imread(img_list[0]).shape

    imgoutdir = os.path.join(os.path.dirname(os.path.dirname(args.indir)), "e2calib_undistorted2")
    os.makedirs(imgoutdir, exist_ok=True)

    pbar = tqdm.tqdm(total=len(img_list))
    for i in range(len(img_list)):
        # undistort img
        image = cv2.imread(img_list[i])
        img_undist = cv2.fisheye.undistortImage(image, K_evs, dist_coeffs, Knew=K_new_evs, new_size=(W, H))
        cv2.imwrite(os.path.join(imgoutdir, f"{i:021d}.png"), img_undist)
        pbar.update(1)

if __name__ == "__main__":
    main()
