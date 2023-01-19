from http.cookiejar import LoadError
import os
import glob
import tqdm
import math
import random
import warnings
import tensorboardX

import numpy as np
import pandas as pd

import time
from datetime import datetime

import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

import struct
import imageio
import trimesh
import mcubes
from rich.console import Console
from torch_ema import ExponentialMovingAverage
import shutil

from utils.plot_utils import *
from utils.event_utils import *

from packaging import version as pver

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
    # :p, gt: torch.Tenosr(H, W, C) on cpu
    # prepare inputs: lipips needs (1, C, H, W)
    gt_lpips = (2.*(gt)-1.)[None,...].permute([0, 3, 1, 2])
    p_lpips = (2.*(p)-1)[None,...].permute([0, 3, 1, 2])

    if rgb_channels == 1:
        _, _, H, W = gt_lpips.shape
        gt_lpips = gt_lpips.expand(1, 3, H, W)
        p_lpips = p_lpips.expand(1, 3, H, W)

    lpips_alex = loss_fn_alex(gt_lpips, p_lpips).numpy()[0][0][0][0]
    lpips_vgg = loss_fn_vgg(gt_lpips, p_lpips).numpy()[0][0][0][0]

    return lpips_alex, lpips_vgg

def compute_pnsr(img0, img1, max_val=255):
    assert max_val > 0.00000001
    assert img0.shape == img1.shape
    return -10 * np.log10(np.mean(np.power(img0.astype(np.float32) - img1.astype(np.float32), 2))) + 20 * np.log10(max_val)


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

@torch.jit.script
def linear_to_srgb(x):
    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x ** 0.41666 - 0.055)

@torch.jit.script
def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, error_map=None):
    ''' get rays
    Args:
        N: sampling N rays. N<0 to sample all pixels.
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = custom_meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device))
    i = i.t().reshape([1, H*W]).expand([B, H*W]) 
    j = j.t().reshape([1, H*W]).expand([B, H*W])

    results = {}

    if N > 0:
        N = min(N, H*W)

        if error_map is None:
            inds = torch.randint(0, H*W, size=[N], device=device) # may duplicate
            inds = inds.expand([B, N])
        else:
            # weighted sample on a low-reso grid
            inds_coarse = torch.multinomial(error_map.to(device), N, replacement=False) # [B, N], but in [0, 128*128)

            # map to the original resolution with random perturb.
            inds_x, inds_y = inds_coarse // 128, inds_coarse % 128 # `//` will throw a warning in torch 1.10... anyway.
            sx, sy = H / 128, W / 128
            inds_x = (inds_x * sx + torch.rand(B, N, device=device) * sx).long().clamp(max=H - 1)
            inds_y = (inds_y * sy + torch.rand(B, N, device=device) * sy).long().clamp(max=W - 1)
            inds = inds_x * W + inds_y

            results['inds_coarse'] = inds_coarse # need this when updating error_map

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['inds'] = inds

    else:
        inds = torch.arange(H*W, device=device).expand([B, H*W])

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)  # (B, N, 3)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True) # (B, N, 3)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2) # (B, N, 3) @ (B, 3, 3) => (B, N_rays, 3)

    rays_o = poses[..., :3, 3] # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d) # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d

    return results

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


@torch.cuda.amp.autocast(enabled=False)
def get_event_rays(xs, ys, c2w_before, c2w_at, intrinsics):
    """
    Inputs:
    xs,ys: torch.Tensor [Nevs]
    c2w_before: [B, Nevs, 3, 4] pose for each event
    c2w_at: [B, Nevs, 3, 4] pose at the event

    Returns:
    rays_o, rays_d: [B, Nevs, 3]
    """
    rays_evs = {}
    fx, fy, cx, cy = intrinsics

    # Get event origins
    rays_o1 = c2w_before[..., :3, 3] # (B, Nevs, 3)
    rays_o2 = c2w_at[..., :3, 3] # (B, Nevs, 3)

    # Get Event Pixel coordinates for unprojection (poses_ev is right, down, front)
    zs = torch.ones_like(xs)
    us = (xs - cx) / fx * zs
    vs = (ys - cy) / fy * zs
    dirs_cams = torch.stack((us, vs, zs), dim=-1) # (Nevs, 3)
    dirs_cams = dirs_cams / torch.norm(dirs_cams, dim=-1, keepdim=True)

    rays_d1 = torch.sum(dirs_cams[..., None, :] * c2w_before[..., :3, :3], axis=-1) # sum((B, Nevs, 1, 3) * (B, Nevs, 3, 3)) =>  (B, Nevs, 3)
    rays_d2 = torch.sum(dirs_cams[..., None, :] * c2w_at[..., :3, :3], axis=-1)
    
    rays_evs["rays_evs_o1"] = rays_o1
    rays_evs["rays_evs_d1"] = rays_d1
    rays_evs["rays_evs_o2"] = rays_o2
    rays_evs["rays_evs_d2"] = rays_d2
    return rays_evs


def extract_fields(bound_min, bound_max, resolution, query_func, S=128):

    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(S)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(S)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(S)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [S, 3]
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy() # [S, 1] --> [x, y, z]
                    u[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys), zi * S: zi * S + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    #print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)

    #print(u.shape, u.max(), u.min(), np.percentile(u, 50))
    
    vertices, triangles = mcubes.marching_cubes(u, threshold)

    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


class PSNRMeter:
    def __init__(self, opt, select_frames):
        self.V = 0
        self.N = 0
        self.use_events = opt.events
        self.select_frames = select_frames

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        for p, t in zip(preds, truths):
            p, t = self.prepare_inputs(p, t) # [B, N, 3] or [B, H, W, 3], range[0, 1]
            psnr = -10 * np.log10(np.mean(np.power(p - t, 2))) #  max_pixel_value == 1
            
            self.V += psnr
            self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "PSNR"), self.measure(), global_step)

    def report(self):
        return f'PSNR = {self.measure():.6f}'

class Trainer(object):
    def __init__(self,
                 name, # name of this experiment
                 opt, # extra conf
                 model, # network 
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 max_keep_ckpt=2, # max num of saved ckpts in disk
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metric
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step 
                 ):
        
        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = opt.eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()
        # event-related
        self.use_events = opt.events
        self.use_luma = opt.use_luma
        self.C_thres = opt.C_thres
        self.event_only = opt.event_only
        self.accumulate_evs = opt.accumulate_evs
        self.log_implicit_C_thres = opt.log_implicit_C_thres
        self.eval_stereo_views = opt.eval_stereo_views
        self.negative_event_sampling = opt.negative_event_sampling
        self.out_dim_color = opt.out_dim_color
        self.epoch_start_noEvLoss = opt.epoch_start_noEvLoss
        self.render_mode = opt.render_mode
        self.weight_loss_rgb = opt.weight_loss_rgb
        self.w_no_ev = opt.w_no_ev
        self.linlog = opt.linlog
        if not self.linlog:
            self.log_thres = torch.Tensor([0.0000001]).to(self.device)

        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4) # naive adam
        else:
            self.optimizer = optimizer(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 1
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
            }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        conffile = os.path.basename(opt.config)
        p, upfolder = os.path.split(os.path.dirname(opt.config))
        upupfolder = os.path.split(p)[1]
        expname = os.path.join(opt.expweek, opt.expname, upupfolder, upfolder+"_"+conffile[:-4])
        self.workspace = os.path.join(opt.outdir, expname)
        print(f"Logging results to {self.workspace}")

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None and not opt.render_mode:
            os.makedirs(self.workspace, exist_ok=True)        
            self.log_path = os.path.join(self.workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)
        elif opt.render_mode:
            self.workspace = opt.model_dir 
            self.ckpt_path = os.path.join(self.workspace, 'checkpoints/')
            
        self.slurm_id = str(os.environ.get("SLURM_JOBID"))
        self.process_id = str(os.getpid())
        # Copying all args to args_slurmID_PID.txt
        args_outpath = os.path.join(self.workspace, "args_" + self.slurm_id + "_" + self.process_id + ".txt")
        if not opt.render_mode:
            with open(args_outpath, "w") as file:
                for arg in sorted(vars(opt)):
                    attr = getattr(opt, arg)
                    file.write("{} = {}\n".format(arg, attr))
        
        # Copy config file (redundant, but better overview than args)
        if opt.config is not None and not opt.render_mode:
            f = os.path.join(self.workspace, "config_" + self.slurm_id + "_" + self.process_id + ".txt")
            with open(f, "w") as file:
                file.write(open(opt.config, "r").read())

        if not opt.render_mode:
            f = open(os.path.join(self.workspace, "calling_code.txt"), "w")
            code_dir = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
            f.write(f"{code_dir}") 
            f.close()
        
        debug = False
        if sys.gettrace() is not None:
            debug = True

        if debug == False and not opt.render_mode:
            codePath = os.path.join(self.workspace, "code_" + self.slurm_id + "_" + self.process_id + "/")
            print(f"Copying this code from {code_dir} to {codePath}")
            shutil.copytree(code_dir, codePath, symlinks=True, dirs_exist_ok=True)

        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

    def __del__(self):
        if self.log_ptr: 
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute: 
                #print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr: 
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file

    ### ------------------------------	
    def train_step_events(self, data):
        images = data["images"] # [B, N, 3]
        B, N, C = images.shape
        loss_evs, loss_no_evs, loss_frames = -1,-1,-1  # init for logging

        bg_color_evs = torch.rand((B, 1, self.out_dim_color)).to(self.device) # (B, Nevs, 3)
        outputs1 = self.model.render(data["rays_evs_o1"], data["rays_evs_d1"], staged=False, bg_color=bg_color_evs, perturb=True, **vars(self.opt))
        outputs2 = self.model.render(data["rays_evs_o2"], data["rays_evs_d2"], staged=False, bg_color=bg_color_evs, perturb=True, **vars(self.opt))

        # Convert I => log(I)
        if self.use_luma:
            pred_luma1 = rgb_to_luma(outputs1["image"], esim=True) # (B, Nevs, 1)
            pred_luma2 = rgb_to_luma(outputs2["image"], esim=True) # (B, Nevs, 1)
            if self.linlog:
                pred_linlog1 = lin_log(pred_luma1*255, linlog_thres=20) # (B, Nevs, 1)
                pred_linlog2 = lin_log(pred_luma2*255, linlog_thres=20) # (B, Nevs, 1)
            else:
                pred_linlog1 = torch.log(torch.maximum(pred_luma1*255, self.log_thres))
                pred_linlog2 = torch.log(torch.maximum(pred_luma1*255, self.log_thres))
        else:
            if self.linlog:
                pred_linlog1 = lin_log(outputs1["image"]*255, linlog_thres=20) # (B, Nevs, 3)
                pred_linlog2 = lin_log(outputs2["image"]*255, linlog_thres=20) # (B, Nevs, 3)
            else:
                pred_linlog1 = torch.log(torch.maximum(outputs1["image"]*255, self.log_thres))
                pred_linlog2 = torch.log(torch.maximum(outputs2["image"]*255, self.log_thres))

        # Compute L2-L1 and pols
        w_evLoss = 1
        delta_linlog = (pred_linlog2 - pred_linlog1) # (B, Nevs, 1or3)
        gt_pol = (data["pols"][..., None]) # (B, Nevs, 1)
        est_C_thres = None
        if self.log_implicit_C_thres:
            est_C_thres = estimate_C_thres_from_pol_dL(gt_pol, delta_linlog, esim=True) # for debugging

        # Compute Loss
        if (self.C_thres != -1):
            # sometimes torch.abs() looks better than **2, e.g. for rgb+events on Shake Carpet 1
            loss_evs = w_evLoss * torch.mean((delta_linlog - gt_pol * self.C_thres)**2) 
        else:
            EPS = 1e-9
            w_evLoss *= 20 # larger weight for better comparability
            if not self.event_only:
                w_evLoss *= 20 # seems to help for normalized loss
            delta_linlog_normed = delta_linlog / (torch.linalg.norm(delta_linlog, dim=1, keepdim=True) + EPS)
            sum_pol_normed = gt_pol / (torch.linalg.norm(gt_pol, dim=1, keepdim=True) + EPS)
            loss_evs = w_evLoss * torch.mean((delta_linlog_normed - sum_pol_normed)**2)
        
        loss = loss_evs
        if not self.event_only:
            rays_o = data['rays_o'] # [B, N, 3]
            rays_d = data['rays_d'] # [B, N, 3]

            # train with random background color if using alpha mixing
            if C == 4:
                bg_color = torch.rand_like(images[..., :self.out_dim_color]) # [B, N, 3], pixel-wise random.
                gt_rgb = images[..., :self.out_dim_color] * images[..., self.out_dim_color:] + bg_color * (1 - images[..., self.out_dim_color:])
            else:
                bg_color = None
                gt_rgb = images

            outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=bg_color, perturb=True, **vars(self.opt))
            pred_rgb = outputs['image']
            loss_frames = self.criterion(pred_rgb, gt_rgb).mean()
            loss = loss + self.weight_loss_rgb * loss_frames

        if self.negative_event_sampling and self.epoch > self.epoch_start_noEvLoss:
            bg_color_evs = torch.rand((B, 1, self.out_dim_color)).to(self.device) # (B, Nnoevs, 3)
            outputs1 = self.model.render(data["rays_no_evs_o1"], data["rays_no_evs_d1"], staged=False, bg_color=bg_color_evs, perturb=True, **vars(self.opt))
            outputs2 = self.model.render(data["rays_no_evs_o2"], data["rays_no_evs_d2"], staged=False, bg_color=bg_color_evs, perturb=True, **vars(self.opt))

            # Convert I => log(I)
            if self.use_luma:
                pred_luma1 = rgb_to_luma(outputs1["image"], esim=True) # (B, Nnoevs, 1)
                pred_linlog1 = lin_log(pred_luma1*255, linlog_thres=20) # (B, Nnoevs, 1)

                pred_luma2 = rgb_to_luma(outputs2["image"], esim=True) # (B, Nnoevs, 1)
                pred_linlog2 = lin_log(pred_luma2*255, linlog_thres=20) # (B, Nnoevs, 1)
            else:
                pred_linlog1 = lin_log(outputs1["image"]*255, linlog_thres=20) # (B, Nnoevs, 3)
                pred_linlog2 = lin_log(outputs2["image"]*255, linlog_thres=20) # (B, Nnoevs, 3)

            abs_diff = torch.abs(pred_linlog2 - pred_linlog1) # (B, Nnoevs, 1or3)
            Cno = self.C_thres if self.C_thres > 0 else 0.25 
            loss_no_evs = self.w_no_ev * torch.mean(torch.relu(abs_diff - Cno))
            loss = loss + loss_no_evs

        losses = {}
        losses["loss_evs"] = loss_evs
        losses["loss_no_evs"] = loss_no_evs
        losses["loss_frames"] = loss_frames
        return delta_linlog, gt_pol, loss, est_C_thres, losses

    def train_step(self, data):
        
        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        images = data['images'] # [B, N, 3/4]

        B, N, C = images.shape

        if self.opt.color_space == 'linear':
            images[..., :self.out_dim_color] = srgb_to_linear(images[..., :self.out_dim_color])
    
        if self.model.bg_radius > 0:
            bg_color = 1
        # train with random background color if not using a bg model.
        else:
            # did not find huge difference with different backgrounds for E-Nerf
            #bg_color = torch.ones(3, device=self.device) # [3], fixed white background
            #bg_color = torch.rand(3, device=self.device) # [3], frame-wise random.
            bg_color = torch.rand_like(images[..., :self.out_dim_color]) # [N, 3], pixel-wise random.

        if C == 4:
            gt_rgb = images[..., :self.out_dim_color] * images[..., self.out_dim_color:] + bg_color * (1 - images[..., self.out_dim_color:])
        else:
            gt_rgb = images

        outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=bg_color, perturb=True, force_all_rays=False, **vars(self.opt))
    
        pred_rgb = outputs['image']

        loss = self.criterion(pred_rgb, gt_rgb).mean(-1) # [B, N, 3] --> [B, N]

        # special case for CCNeRF's rank-residual training
        if len(loss.shape) == 3: # [K, B, N]
            loss = loss.mean(0)

        # update error_map
        if self.error_map is not None:
            index = data['index'] # [B]
            inds = data['inds_coarse'] # [B, N]

            # take out, this is an advanced indexing and the copy is unavoidable.
            error_map = self.error_map[index] # [B, H * W]

            # [debug] uncomment to save and visualize error map
            # if self.global_step % 1001 == 0:
            #     tmp = error_map[0].view(128, 128).cpu().numpy()
            #     print(f'[write error map] {tmp.shape} {tmp.min()} ~ {tmp.max()}')
            #     tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
            #     cv2.imwrite(os.path.join(self.workspace, f'{self.global_step}.jpg'), (tmp * 255).astype(np.uint8))

            error = loss.detach().to(error_map.device) # [B, N], already in [0, 1]
            
            # ema update
            ema_error = 0.1 * error_map.gather(1, inds) + 0.9 * error
            error_map.scatter_(1, inds, ema_error)

            # put back
            self.error_map[index] = error_map

        loss = loss.mean()

        return pred_rgb, gt_rgb, loss

    def eval_step(self, data):
        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        images = data['images'] # [B, H, W, 3/4]

        B, H, W, C = images.shape
        # C = self.out_dim_color

        if self.opt.color_space == 'linear':
            images[..., :self.out_dim_color] = srgb_to_linear(images[..., :self.out_dim_color])

        # eval with fixed background color
        bg_color = 1
        if C == 4:
            gt_rgb = images[..., :self.out_dim_color] * images[..., self.out_dim_color:] + bg_color * (1 - images[..., self.out_dim_color:])
        else:
            gt_rgb = images
        
        outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=False, **vars(self.opt))

        pred_rgb = outputs['image'].reshape(B, H, W, self.out_dim_color)
        pred_depth = outputs['depth'].reshape(B, H, W)

        loss = self.criterion(pred_rgb, gt_rgb).mean()

        return pred_rgb, pred_depth, gt_rgb, loss

    def eval_step_tumvie(self, data, loader):
        pred_rgb0, pred_depth0, gt_rgb0, loss0 = self.eval_step(data)

        rays_o = data['rays_evs_o'] # [B, N, 3]
        rays_d = data['rays_evs_d'] # [B, N, 3]
        images = data['images'] # [B, H, W, 3/4]
        H, W = loader._data.H_ev, loader._data.W_ev
        B, C = 1, self.out_dim_color

        if self.opt.color_space == 'linear':
            images[..., :self.out_dim_color] = srgb_to_linear(images[..., :self.out_dim_color])

        bg_color = 1
        if C == 4:
            gt_rgb = images[..., :self.out_dim_color] * images[..., self.out_dim_color:] + bg_color * (1 - images[..., self.out_dim_color:])
        else:
            gt_rgb = images
        
        outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=False, **vars(self.opt))
        pred_rgb = outputs['image'].reshape(B, H, W, self.out_dim_color)
        pred_depth = outputs['depth'].reshape(B, H, W)
        
        loss = loss0 
        pred_rgbs = [pred_rgb0.squeeze(0), pred_rgb.squeeze(0)]
        pred_depth = [pred_depth0.squeeze(0), pred_depth.squeeze(0)]
        gt_rgb = [gt_rgb0.squeeze(0), pred_rgb.squeeze(0)]

        return pred_rgbs, pred_depth, gt_rgb, loss

    # moved out bg_color and perturb for more flexible control...
    def test_step(self, data, bg_color=None, perturb=False):  

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        H, W = data['H'], data['W']

        if bg_color is not None:
            bg_color = bg_color.to(self.device)

        outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=perturb, **vars(self.opt))

        pred_rgb = outputs['image'].reshape(-1, H, W, self.out_dim_color)
        pred_depth = outputs['depth'].reshape(-1, H, W)

        return pred_rgb, pred_depth


    def save_mesh(self, save_path=None, resolution=256, threshold=10):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'meshes', f'{self.name}_{self.epoch}.ply')

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        def query_func(pts):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    sigma = self.model.density(pts.to(self.device))['sigma']
            return sigma

        vertices, triangles = extract_geometry(self.model.aabb_infer[:3], self.model.aabb_infer[3:], resolution=resolution, threshold=threshold, query_func=query_func)

        mesh = trimesh.Trimesh(vertices, triangles, process=False) # important, process=True leads to seg fault...
        mesh.export(save_path)

        self.log(f"==> Finished saving mesh.")

    def train(self, train_loader, valid_loader, max_epochs):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        # mark untrained region (i.e., not covered by any camera from the training dataset)
        if self.model.cuda_ray:
            self.model.mark_untrained_grid(train_loader._data.poses, train_loader._data.intrinsics)

        # get a ref to error_map
        self.error_map = train_loader._data.error_map
        
        dt_eps = 0
        dt_logeps = 0
        for epoch in range(self.epoch, max_epochs + 1):
            self.epoch = epoch
            
            self.train_one_epoch(train_loader)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)


            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader, save_path=None, name=None):
        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)
        os.makedirs(os.path.join(save_path, "depth"), exist_ok=True)
        
        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(loader):
                
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth = self.test_step(data)                
                
                path = os.path.join(save_path, f'{name}_{i:04d}.png')

                if self.opt.color_space == 'linear':
                    preds = linear_to_srgb(preds)

                pred = preds[0].detach().cpu().numpy()
            
                cv2.imwrite(path, cv2.cvtColor((pred * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                
                if self.epoch % 100 == 0: # save depth less often
                    path_depth = os.path.join(save_path, "depth", f'{name}_{i:04d}_depth.png')
                    pred_depth = preds_depth[0].detach().cpu().numpy()
                    cv2.imwrite(path_depth, (pred_depth * 255).astype(np.uint8))

                pbar.update(loader.batch_size)

        self.log(f"==> Finished Test.")
    
    # [GUI] just train for 16 steps, without any other overhead that may slow down rendering.
    def train_gui(self, train_loader, step=16):

        self.model.train()

        total_loss = torch.tensor([0], dtype=torch.float32, device=self.device)
        
        loader = iter(train_loader)

        for _ in range(step):
            
            # mimic an infinite loop dataloader (in case the total dataset is smaller than step)
            try:
                data = next(loader)
            except StopIteration:
                loader = iter(train_loader)
                data = next(loader)

            # mark untrained grid
            if self.global_step == 0:
                self.model.mark_untrained_grid(train_loader._data.poses, train_loader._data.intrinsics)
                self.error_map = train_loader._data.error_map

            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % 16 == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()
            
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss = self.train_step(data)
         
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            total_loss += loss.detach()

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss.item() / step

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        outputs = {
            'loss': average_loss,
            'lr': self.optimizer.param_groups[0]['lr'],
        }
        
        return outputs

    
    # [GUI] test on a single image
    def test_gui(self, pose, intrinsics, W, H, bg_color=None, spp=1, downscale=1):
        
        # render resolution (may need downscale to for better frame rate)
        rH = int(H * downscale)
        rW = int(W * downscale)
        intrinsics = intrinsics * downscale

        pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)

        rays = get_rays(pose, intrinsics, rH, rW, -1)

        data = {
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'H': rH,
            'W': rW,
        }
        
        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                # here spp is used as perturb random seed!
                preds, preds_depth = self.test_step(data, bg_color=bg_color, perturb=spp)

        if self.ema is not None:
            self.ema.restore()

        # interpolation to the original resolution
        if downscale != 1:
            preds = F.interpolate(preds.permute(0, 3, 1, 2), size=(H, W), mode='nearest').permute(0, 2, 3, 1).contiguous()
            preds_depth = F.interpolate(preds_depth.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)

        if self.opt.color_space == 'linear':
            preds = linear_to_srgb(preds)

        pred = preds[0].detach().cpu().numpy()
        pred_depth = preds_depth[0].detach().cpu().numpy()

        outputs = {
            'image': pred,
            'depth': pred_depth,
        }

        return outputs

    def train_one_epoch(self, loader):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)
        
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        dtlog_steps = 0
        dtsteps, dtloadall, dtloaddisk, dtinterpols, dtrays, dtimgs, dtbundle = 0, 0, 0, 0, 0, 0, 0

        for data in loader:
            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % 16 == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()
                    
            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                if self.use_events:
                    preds, truths, loss, est_C_thres, losses_indiv = self.train_step_events(data)
                else:
                    preds, truths, loss = self.train_step(data)
         
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)
                        
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/epoch", self.epoch, self.global_step)
                    if self.use_events:
                        self.writer.add_scalar("train/loss_evs", losses_indiv["loss_evs"].item(), self.global_step)
                        if self.negative_event_sampling and self.epoch > self.epoch_start_noEvLoss:
                            self.writer.add_scalar("train/loss_no_evs", losses_indiv["loss_no_evs"].item(), self.global_step)
                        if not self.event_only:
                            self.writer.add_scalar("train/loss_frames", losses_indiv["loss_frames"].item(), self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)
                    if self.log_implicit_C_thres and self.use_events:
                        self.writer.add_scalar("train/est_C_med_on", est_C_thres["median_on"], self.global_step)
                        self.writer.add_scalar("train/est_C_med_off", est_C_thres["median_off"], self.global_step)
                        self.writer.add_scalar("train/est_C_med_on_sign", est_C_thres["median_on_sign"], self.global_step)
                        self.writer.add_scalar("train/est_C_med_off_sign", est_C_thres["median_off_sign"], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                    # self.log(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                    # self.log(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)

                if self.epoch <= self.eval_interval:
                    save_path_gt = os.path.join(self.workspace, "validation", "gt_trainViews", f'{self.local_step-1:04d}_gt.png')
                    if not os.path.isdir(os.path.dirname(save_path_gt)):
                        os.makedirs(os.path.dirname(save_path_gt), exist_ok=True)
                    cv2.imwrite(save_path_gt, cv2.cvtColor((loader._data.images[self.local_step-1].detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        
        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")

    def evaluate_one_epoch(self, loader, name=None):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0
                                
            # for now, we only evaluate the rgb-view
            all_preds, all_gts, all_depths = [], [], [] # for event-only (a,b)-correction
            psnrs, lipss_alex, lipss_vggs, ssims = [], [], [], [] # for image-only metrics
            for data in loader:    
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    if (loader._data.mode == "tumvie" or loader._data.mode == "eds") and self.eval_stereo_views: 
                        preds, preds_depth, truths, loss = self.eval_step_tumvie(data, loader)
                    else:
                        preds, preds_depth, truths, loss = self.eval_step(data)
                all_preds.append(preds)
                all_gts.append(truths)
                all_depths.append(preds_depth)

                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size
                    
                    preds_list = [torch.zeros_like(preds).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    preds_depth_list = [torch.zeros_like(preds_depth).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_depth_list, preds_depth)
                    preds_depth = torch.cat(preds_depth_list, dim=0)

                    truths_list = [torch.zeros_like(truths).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(truths_list, truths)
                    truths = torch.cat(truths_list, dim=0)
                
                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:

                    # update average PSNR-metric
                    for metric in self.metrics:
                        metric.update(preds, truths)

                    if self.event_only:
                        self.local_step += 1
                        # need to do (a,b)-correction for event-only
                        continue 

                    # per-image PSNR. chose truths[0] and preds[0] for rgb-view
                    gt_cpu, pred_cpu = truths[0].detach().cpu(), preds[0].detach().cpu()
                    psnr_ = compute_pnsr(gt_cpu.numpy(), pred_cpu.numpy(), max_val=1)
                    self.writer.add_scalar(f"psnr/{self.local_step}", psnr_, self.global_step)
                    psnrs.append(psnr_)

                    # per-image LPIPS
                    lpips_alex, lpips_vgg = compute_lpips(gt_cpu, pred_cpu, self.out_dim_color)
                    self.writer.add_scalar(f"lpips_alex/{self.local_step}", lpips_alex, self.global_step)
                    self.writer.add_scalar(f"lpips_vgg/{self.local_step}", lpips_vgg, self.global_step)
                    lipss_alex.append(lpips_alex)
                    lipss_vggs.append(lpips_vgg)

                    # per-image SSIM
                    ssim_ = ssim(gt_cpu.numpy()[...,0], pred_cpu.numpy()[...,0], data_range=1)
                    self.writer.add_scalar(f"ssim/{self.local_step}", ssim_, self.global_step)
                    ssims.append(ssim_)
                    
                    if self.opt.color_space == 'linear':
                        preds = linear_to_srgb(preds) # only for saving

                    # paths for rgb, raw, depth, gt
                    save_path = os.path.join(self.workspace, "validation", 'prediction', f'{name}_{self.local_step:04d}.png')
                    save_path_raw = os.path.join(self.workspace, "validation", "raw", f'{name}_{self.local_step:04d}.npy')
                    save_path_depth = os.path.join(self.workspace, "validation", "depth", f'{name}_{self.local_step:04d}_depth.png')
                    save_path_gt = os.path.join(self.workspace, "validation", "gt", f'{name}_{self.local_step:04d}_gt.png')
                    if not os.path.isdir(os.path.dirname(save_path)):
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        os.makedirs(os.path.dirname(save_path_raw), exist_ok=True)
                        os.makedirs(os.path.dirname(save_path_depth), exist_ok=True)
                        os.makedirs(os.path.dirname(save_path_gt), exist_ok=True)

                    if (loader._data.mode == "tumvie" or loader._data.mode == "eds") and self.eval_stereo_views: # Event-View (for tumvie)
                        save_path_ev = os.path.join(self.workspace, "validation", "event_view", 'prediction_ev', f'{name}_{self.local_step:04d}.png')
                        save_path_evs_raw = os.path.join(self.workspace, "validation", "event_view", "raw", f'{name}_{self.local_step:04d}.npy')
                        save_path_depth_ev = os.path.join(self.workspace, "validation", "event_view", "depth_ev", f'{name}_{self.local_step:04d}_depth.png')
                        save_path_gt_ev = os.path.join(self.workspace, "validation", "event_view", "warped_gt_ev", f'{name}_{self.local_step:04d}_gt.png')
                        os.makedirs(os.path.dirname(save_path_ev), exist_ok=True)
                        os.makedirs(os.path.dirname(save_path_depth_ev), exist_ok=True)
                        os.makedirs(os.path.dirname(save_path_evs_raw), exist_ok=True)
                        os.makedirs(os.path.dirname(save_path_gt_ev), exist_ok=True)

                        # gt 
                        cv2.imwrite(save_path_gt_ev, cv2.cvtColor((truths[1].detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                        
                        # prediction in event view
                        pred = preds[1].detach().cpu().numpy()
                        cv2.imwrite(save_path_ev, cv2.cvtColor((pred * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                        np.save(save_path_evs_raw, pred) # saving raw

                        # depth in event view
                        pred_depth = preds_depth[1].detach().cpu().numpy()
                        cv2.imwrite(save_path_depth_ev, (pred_depth * 255).astype(np.uint8))

                    # saving rgb-view prediction
                    pred = preds[0].detach().cpu().numpy()
                    #if self.epoch % (self.eval_interval * 5):
                    np.save(save_path_raw, pred) # saving raw
                    text = "psnr: {:.2f} ".format(psnrs[-1]) + " | lpips: {:.2f} ".format(lipss_alex[-1]) +  " | ssim: {:.2f} ".format(ssims[-1])
                    pred_corrected_img = cv2.putText(pred, text, (0, data["H"]-5),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
                    cv2.imwrite(save_path, cv2.cvtColor((pred_corrected_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

                    if True or self.epoch % 100 == 0:
                        pred_depth = preds_depth[0].detach().cpu().numpy()
                        cv2.imwrite(save_path_depth, (pred_depth * 255).astype(np.uint8))

                    if self.epoch <= self.eval_interval:
                        cv2.imwrite(save_path_gt, cv2.cvtColor((truths[0].detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                    
                    self.local_step += 1
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                    pbar.update(loader.batch_size)
            ### end loop over val images (loader)

            ### event_only: (a,b)-correction requires acces too all images
            if self.event_only:
                # convert preds&gts to logspace
                if self.out_dim_color == 3:
                    preds_logs = [torch.log(255*rgb_to_luma(im[0]) + 1e-3) for im in all_preds] # im[0] to get rgb view
                    imgs_gt_log = [torch.log(255*rgb_to_luma(im[0]) + 1e-3) for im in all_gts]
                else:
                    preds_logs = [torch.log(255*im[0] + 1e-3) for im in all_preds]
                    imgs_gt_log = [torch.log(255*im[0] + 1e-3) for im in all_gts]
                    
                preds_logs = torch.stack(preds_logs) # (len(val_idxs), H, W, C)
                imgs_gt_log = torch.stack(imgs_gt_log)
                a, b = solve_normal_equations(preds_logs, imgs_gt_log)
                self.writer.add_scalar(f"a/", a, self.global_step)
                self.writer.add_scalar(f"b/", b, self.global_step)

                if (loader._data.mode == "tumvie" or loader._data.mode == "eds") and self.eval_stereo_views:
                    if self.out_dim_color == 3:
                        ev_views_logs = [torch.log(255*rgb_to_luma(im[1]) + 1e-9) for im in all_preds]
                    else:
                        ev_views_logs = [torch.log(255*im[1] + 1e-9) for im in all_preds] # im[1] for event view
                    assert len(ev_views_logs) == len(imgs_gt_log)

                psnrs_cor, lipss_alex_cor, lipss_vggs_cor, ssims_cor = [], [], [], [] # image-only metrics
                for j in range(len(imgs_gt_log)):
                    pred_cor_j = torch.exp(preds_logs[j] * a + b).detach().cpu()
                    # per-image PSNR
                    gt_j = 255. * all_gts[j][0].detach().cpu()
                    if self.out_dim_color == 3:
                        gt_j = rgb_to_luma(gt_j)
                    psnr_cor = compute_pnsr(gt_j.numpy(), pred_cor_j.numpy(), max_val=255)
                    psnrs_cor.append(psnr_cor)
                    self.log(f"psnr-corrected = {psnr_cor}")
                    self.writer.add_scalar(f"psnr-corrected/{j}", psnr_cor, self.global_step)

                    # per-image LPIPS
                    lpips_alex_cor, lpips_vgg_cor = compute_lpips(gt_j, pred_cor_j, self.out_dim_color)
                    self.writer.add_scalar(f"lpips_alex/{j}", lpips_alex_cor, self.global_step)
                    self.writer.add_scalar(f"lpips_vgg/{j}", lpips_vgg_cor, self.global_step)
                    lipss_alex_cor.append(lpips_alex_cor)
                    lipss_vggs_cor.append(lpips_vgg_cor)

                    # per-image SSIM
                    ssim_cor = ssim(gt_j.numpy()[...,0], pred_cor_j.numpy()[...,0], data_range=255)
                    self.writer.add_scalar(f"ssim/{j}", ssim_cor, self.global_step)
                    ssims_cor.append(ssim_cor)

                    # Make outfolders
                    save_path_pred = os.path.join(self.workspace, "validation", "prediction_corrected", f'{name}_{j:04d}.png')
                    save_path_raw = os.path.join(self.workspace, "validation", "raw", f'{name}_{j:04d}.npy')
                    save_path_depth = os.path.join(self.workspace, "validation", "depth", f'{name}_{j:04d}_depth.png')
                    save_path_gt = os.path.join(self.workspace, "validation", "gt", f'{name}_{j:04d}_gt.png')
                    if not os.path.isdir(os.path.dirname(save_path_pred)):
                        os.makedirs(os.path.dirname(save_path_pred), exist_ok=True)
                        os.makedirs(os.path.dirname(save_path_raw), exist_ok=True)
                        os.makedirs(os.path.dirname(save_path_depth), exist_ok=True)
                        os.makedirs(os.path.dirname(save_path_gt), exist_ok=True)

                    if (loader._data.mode == "tumvie" or loader._data.mode == "eds") and self.eval_stereo_views: # Event-View (for tumvie)
                        save_path_pred_evs = os.path.join(self.workspace, "validation", "event_view", 'prediction_corrected_ev', f'{name}_{j:04d}.png')
                        save_path_raw_evs = os.path.join(self.workspace, "validation", "event_view", "raw", f'{name}_{j:04d}.npy')
                        save_path_depth_ev = os.path.join(self.workspace, "validation", "event_view", "depth_ev", f'{name}_{j:04d}_depth.png')
                        
                        if not os.path.isdir(os.path.dirname(save_path_pred_evs)):
                            os.makedirs(os.path.dirname(save_path_pred_evs), exist_ok=True)
                            os.makedirs(os.path.dirname(save_path_raw_evs), exist_ok=True)
                            os.makedirs(os.path.dirname(save_path_depth_ev), exist_ok=True)

                        # prediction in event view (can not compute metrics here, only saving renderings)
                        ev_pred_cor_j = torch.exp(ev_views_logs[j] * a + b).detach().cpu()
                        ev_pred_cor_j = np.clip(ev_pred_cor_j.numpy(), a_min=0, a_max=255)
                        ev_pred_cor_j = np.rint(ev_pred_cor_j).astype(np.uint8)
                        cv2.imwrite(save_path_pred_evs, ev_pred_cor_j)
                        np.save(save_path_raw_evs, all_preds[j][1].detach().cpu().numpy()) # raw event view
                        if True or self.epoch % 100 == 0: # depth event view
                            cv2.imwrite(save_path_depth_ev, cv2.cvtColor((all_depths[j][1].detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

                    # saving rgb-view prediction. clip image to range and draw text
                    pred_corrected_img = np.clip(pred_cor_j.numpy(), a_min=0, a_max=255)
                    pred_corrected_img = np.rint(pred_corrected_img).astype(np.uint8)
                    text = "psnr-cor: {:.2f} ".format(psnrs_cor[-1]) + " | lpips: {:.2f} ".format(lipss_alex_cor[-1]) +  " | ssim: {:.2f} ".format(ssims_cor[-1])
                    pred_corrected_img = cv2.putText(pred_corrected_img, text, (0, data["H"]-5),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
                    cv2.imwrite(save_path_pred, pred_corrected_img)
                    # if self.epoch % (self.eval_interval * 5):
                    np.save(save_path_raw, all_preds[j][0].detach().cpu().numpy()) # save raw as well
                    
                    if True or self.epoch % 100 == 0:
                        cv2.imwrite(save_path_depth, (all_depths[j][0].detach().cpu().numpy() * 255).astype(np.uint8))

                    if self.epoch <= self.eval_interval:
                        cv2.imwrite(save_path_gt, cv2.cvtColor((all_gts[j][0].detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                # end loop over j (all_gts)
                self.writer.add_scalar("psnr-corrected/mean", np.asarray(psnrs_cor).mean(), self.global_step)
                self.writer.add_scalar("lpips_vgg/mean", np.asarray(lipss_vggs_cor).mean(), self.global_step)
                self.writer.add_scalar("lpips_alex/mean", np.asarray(lipss_alex_cor).mean(), self.global_step)
                self.writer.add_scalar("ssim/mean", np.asarray(ssims_cor).mean(), self.global_step)
            ### end saving corrected images
            else:
                self.writer.add_scalar("psnr/mean", np.asarray(psnrs).mean(), self.global_step)
                self.writer.add_scalar("lpips_vgg/vgg_mean", np.asarray(lipss_vggs).mean(), self.global_step)
                self.writer.add_scalar("lpips_alex/alex_mean", np.asarray(lipss_alex).mean(), self.global_step)
                self.writer.add_scalar("ssim/ssim_mean", np.asarray(ssims).mean(), self.global_step)

        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(result if self.best_mode == 'min' else - result) # if max mode, use -result
            else:
                self.stats["results"].append(average_loss) # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self, name=None, full=False, best=False, remove_old=True):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }

        if self.model.cuda_ray:
            state['mean_count'] = self.model.mean_count
            state['mean_density'] = self.model.mean_density

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()
        
        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{self.ckpt_path}/{name}.pth"

            if remove_old:
                self.stats["checkpoints"].append(file_path)

                if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                    old_ckpt = self.stats["checkpoints"].pop(0)
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)

            torch.save(state, file_path)

        else:    
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                    self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results 
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    if self.ema is not None:
                        self.ema.restore()
                    
                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")
            
    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/*_ep*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                if self.render_mode:
                    sys.exit()
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        
        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")   

        if self.ema is not None and 'ema' in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict['ema'])

        if self.model.cuda_ray:
            if 'mean_count' in checkpoint_dict:
                self.model.mean_count = checkpoint_dict['mean_count']
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']
        
        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")
        
        if self.optimizer and  'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")
        
        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")
        
        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")
