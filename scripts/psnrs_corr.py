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
import lpips
loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimizatio
from skimage.metrics import structural_similarity as ssim


def solve_normal_equations(preds_logs, imgs_gt_log):
    # :preds_logs: torch.Tensor (N, H, W, 1)
    # :imgs_gt_log torch.Tensor (N, H, W, 1)

    # setup normal equations
    N_imgs = len(imgs_gt_log)
    gt_img_shape = imgs_gt_log[0].shape
    N = N_imgs * gt_img_shape[0] * gt_img_shape[1] * gt_img_shape[2]
    X = np.ones((N, 2))
    X[:, 1] = preds_logs.flatten().detach().cpu().numpy()
    y = imgs_gt_log.flatten().detach().cpu().numpy()

    # solve normal equaiton
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    a = beta[1]
    b = beta[0]

    if np.isnan(b):
        b = 5
    elif np.isnan(-b):
        b = -5

    if np.isnan(a):
        a = 5
    elif np.isnan(-a):
        a = -5

    return a, b

def compute_lpips(p, gt, rgb_channels=3):
    # :p, gt: torch.Tenosr(H, W, C) on cpu (fails on gpu)
    # prepare inputs: lipips needs (1, C, H, W)
    gt_lpips = (2.*(gt)-1.)[None,...].permute([0, 3, 1, 2])
    p_lpips = (2.*(p)-1)[None,...].permute([0, 3, 1, 2])

    if rgb_channels == 1:
        _, _, H, W = gt_lpips.shape
        gt_lpips = gt_lpips.expand(1, 3, H, W)
        p_lpips = p_lpips.expand(1, 3, H, W)

    lpips_alex = loss_fn_alex(gt_lpips, p_lpips).detach().cpu().numpy()[0][0][0][0]
    lpips_vgg = loss_fn_vgg(gt_lpips, p_lpips).detach().cpu().numpy()[0][0][0][0]

    return lpips_alex, lpips_vgg

def compute_pnsr(img0, img1, max_val=255):
    assert max_val > 0.00000001
    assert img0.shape == img1.shape
    return -10 * np.log10(np.mean(np.power(img0.astype(np.float32) - img1.astype(np.float32), 2))) + 20 * np.log10(max_val)

def main():
    parser = argparse.ArgumentParser(description="Raw to png images in dir")
    parser.add_argument(                               
        "--indir", help="Input raw dir.", \
            default="../raws"
    )
    parser.add_argument(                               
        "--start_from", help="Start idx", default=0, type=int
    )
    parser.add_argument(                               
        "--event_only", help="event_only", default=1, type=int
    )
    parser.add_argument(                               
        "--cut", help="cut psnr", default=1, type=int
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdim_color = 1

    cut_h = [400, 600]
    cut_w = [100, 900]

    args = parser.parse_args()
    indir = args.indir
    outdir = os.path.join(os.path.dirname(indir), "corrected_psnr")
    outdirc = os.path.join(outdir, "contrast_spread")
    assert "raw" in indir
    #if "both" in indir or "Both" in indir:
    #    assert args.event_only == 0

    print(f"\nMetrics on {indir}")

    rawfiles_names = sorted(glob.glob(os.path.join(indir, "*.npy")))[args.start_from:]
    # gtdir = os.path.join(os.path.dirname(indir), "gt")
    gts_names = sorted(glob.glob(os.path.join(os.path.dirname(os.path.dirname(indir)), "gt", "*.png")))[args.start_from:]
    assert len(gts_names) == len(rawfiles_names)

    if not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)
    os.makedirs(outdirc, exist_ok=True)

    preds_logs = []
    for i, raw_name in enumerate(rawfiles_names):
        raw_fname = os.path.split(raw_name)[1]
        raw = torch.from_numpy(np.load(raw_name)).to(device)
        if args.cut:
            raw = raw[cut_h[0]:cut_h[1], cut_w[0]:cut_w[1], :]
        if args.event_only:
            preds_logs.append(torch.log(255*rgb_to_luma(raw) + 1e-3))
        else:
            preds_logs.append(255*rgb_to_luma(raw))

    imgs_gts_log, all_gts = [], []
    for i, name in enumerate(gts_names):
        gt = torch.from_numpy(cv2.imread(name)).to(device)
        if args.cut:
            gt = gt[cut_h[0]:cut_h[1], cut_w[0]:cut_w[1], :]
        if args.event_only:
            imgs_gts_log.append(torch.log(rgb_to_luma(gt) + 1e-3))
        else:
            imgs_gts_log.append(rgb_to_luma(gt))
        all_gts.append(gt)

    if args.event_only:
        preds_logs = torch.stack(preds_logs) 
        imgs_gts_log = torch.stack(imgs_gts_log)
        a, b = solve_normal_equations(preds_logs, imgs_gts_log)
        print(f"a  = {a}, b ={b}")

    psnrs_cor, lipss_alex_cor, lipss_vggs_cor, ssims_cor = [], [], [], []
    for j in range(len(imgs_gts_log)):
        #if j < 12:
        #    continue
        if args.event_only:
            pred_cor_j = torch.exp(preds_logs[j] * a + b).detach().cpu()
        else:
            pred_cor_j = preds_logs[j].detach().cpu()
        pred_corrected_img = np.clip(pred_cor_j.numpy(), a_min=0, a_max=255)
        pred_corrected_img = np.rint(pred_corrected_img).astype(np.uint8)
        cv2.imwrite(os.path.join(outdir, f"{j}.png"), pred_corrected_img)
        imgcorspread = (pred_corrected_img - np.min(pred_corrected_img)) / (np.max(pred_corrected_img) - np.min(pred_corrected_img)) * 255
        cv2.imwrite(os.path.join(outdirc, raw_fname[:-4] + f"{j}_spread.png"), imgcorspread)

        # per-image PSNR
        gt_j = all_gts[j].detach().cpu()
        if outdim_color == 3:
            gt_j = rgb_to_luma(gt_j)
        gt_j = rgb_to_luma(gt_j)

        psnr_cor = compute_pnsr(gt_j.numpy(), pred_cor_j.numpy(), max_val=255)
        psnrs_cor.append(psnr_cor)

        # per-image LPIPS
        lpips_alex_cor, lpips_vgg_cor = compute_lpips(gt_j, pred_cor_j, outdim_color)
        lipss_alex_cor.append(lpips_alex_cor)
        lipss_vggs_cor.append(lpips_vgg_cor)

        # per-image SSIM
        ssim_cor = ssim(gt_j.numpy()[...,0], pred_cor_j.numpy()[...,0], data_range=255)
        ssims_cor.append(ssim_cor)
        print(f"{j}: PSNR-cor: {psnr_cor}. LP-alex-cor: {lpips_alex_cor}. SSIMS-cor: {ssim_cor}")
        print(f"{j}: {psnr_cor:.2f} & {lpips_alex_cor:.2f} & {ssim_cor:.2f}\n")


    print(f"Mean values: PSNR-cor: {np.array(psnrs_cor).mean()}. SSIMS-cor: {np.array(ssims_cor).mean()}. LP-alex-cor: {np.array(lipss_alex_cor).mean()}.")
    print(f"{np.array(psnrs_cor).mean():.3f} & {np.array(ssims_cor).mean():.3f} & {np.array(lipss_alex_cor).mean():.3f}\n\n")

    



if __name__ == "__main__":
    main()
