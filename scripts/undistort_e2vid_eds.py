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
        "--indir", help="Input image directory.", default="DATADIR/e2vids/left/e2vid_up4_freq0/e2calib/"
    )
    parser.add_argument(                               
        "--calibstr", help="Start idx", default="calib0", type=str
    )
    args = parser.parse_args()

    print(f"Undistorting {args.indir}")

    assert "e2vid" in args.indir or "e2calib" in args.indir
    calibstr = args.calibstr
    assert calibstr == "calib1" or calibstr == "calib0"

    K_evs = np.zeros((3,3))
    if calibstr == "calib1":
        K_evs[0,0] = 548.8989250692618
        K_evs[0,2] = 313.5293514832678
        K_evs[1,1] = 550.0282089284915
        K_evs[1,2] = 219.6325753720951
        K_evs[2, 2] = 1
        dist_coeffs_evs = np.asarray([-0.08095806072593555, 0.15743578875760092, -0.0035154416164982195, -0.003950567808338846])
    elif calibstr == "calib0":
        K_evs[0,0] = 560.8520948927032
        K_evs[0,2] = 313.00733235019237
        K_evs[1,1] = 560.6295819972383
        K_evs[1,2] = 217.32858679842997
        K_evs[2, 2] = 1
        dist_coeffs_evs = np.asarray([-0.09776467241921379, 0.2143738428636279, -0.004710710105172864, -0.004215916089401789])

    W, H = 640, 480
    K_new_evs, roi = cv2.getOptimalNewCameraMatrix(K_evs, dist_coeffs_evs, (W, H), alpha=0, newImgSize=(W, H))
    x,y,w,h = roi
    ev_mapx, ev_mapy = cv2.initUndistortRectifyMap(K_evs, dist_coeffs_evs, np.eye(3), K_new_evs, (W, H), cv2.CV_32FC1)

    img_list = sorted(glob.glob(os.path.join(args.indir) + "*.png"))
    img_list = sorted([os.path.join(args.indir, im) for im in img_list if im.endswith(".png")])
    H, W, _ = cv2.imread(img_list[0]).shape

    imgoutdir = os.path.join(os.path.dirname(os.path.dirname(args.indir)), "e2calib_undistorted2")
    os.makedirs(imgoutdir, exist_ok=True)
    
    pbar = tqdm.tqdm(total=len(img_list))
    for i in range(len(img_list)):
        # undistort img
        image =  cv2.imread(img_list[i])
        img_undist = cv2.undistort(image, K_evs, dist_coeffs_evs, newCameraMatrix=K_new_evs) # k1,k2,p1,p2
        img_undist2 = cv2.remap(image, ev_mapx, ev_mapy, cv2.INTER_LINEAR)  # 
        assert compute_psnr(img_undist2, img_undist) > 50
        cv2.imwrite(os.path.join(imgoutdir, f"{i:021d}.png"), img_undist)
        pbar.update(1)

if __name__ == "__main__":
    main()
