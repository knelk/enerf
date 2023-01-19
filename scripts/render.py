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
from nerf.provider import NeRFDataset, EventNeRFDataset
from nerf.utils import Trainer, PSNRMeter
from scipy.spatial.transform import Slerp, Rotation

from nerf.utils import get_rays
from utils.pose_utils import *
from utils.plot_utils import *
import pandas as pd
import configargparse
import random
import os
import trimesh
"""
Descriptions:
This script plots the camera&render path and renders images from trained nerf model.
"""

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_model(opt): 
    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True
        opt.preload = True

    if opt.ff:
        opt.fp16 = True
        assert opt.bg_radius <= 0, "background model is not implemented for --ff"
        from nerf.network_ff import NeRFNetwork
    elif opt.tcnn:
        opt.fp16 = True
        assert opt.bg_radius <= 0, "background model is not implemented for --tcnn"
        from nerf.network_tcnn import NeRFNetwork
    else:
        from nerf.network import NeRFNetwork

    print(opt)
    seed_everything(opt.seed)

    model = NeRFNetwork(
        encoding="hashgrid",
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=opt.density_scale,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
        disable_view_direction=opt.disable_view_direction,
        out_dim_color=opt.out_dim_color
    )
    model_params = list(model.sigma_net.parameters()) + list(model.color_net.parameters())
    encoding_params = list(model.encoder.parameters())

    print(model)
    return model, model_params, encoding_params


def parse_cfg_file(cfgfile):
    parser = configargparse.ArgumentParser()

    parser.add_argument(                       
        "--model_dir", help="Directory to trained model. Must include transforms.jsonto render randomly around training trajectory", 
        type=str,
        default="EXPDIR"
    )
    parser.add_argument("--config", default=cfgfile, is_config_file=True, help="config file path")

    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --preload")
    parser.add_argument('--outdir', type=str, default="OUTDIR")
    parser.add_argument('--expweek', type=str, default="testweek")
    parser.add_argument('--expname', type=str, default="testname")
    parser.add_argument('--datadir', type=str, default="DATADIR")  # 
    parser.add_argument('--train_idxs', type=int, action="append")
    parser.add_argument('--val_idxs', type=int, action="append")
    parser.add_argument('--test_idxs', type=int, action="append")
    parser.add_argument('--exclude_idxs', type=int, action="append")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--disable_view_direction', type=int, default=0)
    parser.add_argument('--out_dim_color', type=int, default=1)
    # Event-Related
    parser.add_argument('--hotpixs', type=int, default=0)
    parser.add_argument('--e2vid', type=int, default=0, help="select upsample factor with this value")
    parser.add_argument('--events', type=int, default=0)
    parser.add_argument('--event_only', type=int, default=0)
    parser.add_argument('--accumulate_evs', type=int, default=0)
    parser.add_argument('--acc_max_num_evs', type=int, default=0, help="max num successors for event accumulation. if 0: use all, if > 0: use up max_num (random)")
    parser.add_argument('--use_luma', type=int, default=1)
    parser.add_argument('--linlog', type=int, default=1)
    parser.add_argument('--batch_size_evs', type=int, default=4096)
    parser.add_argument('--C_thres', type=float, default=0.5)
    parser.add_argument('--images_corrupted', type=int, default=0)
    parser.add_argument('--log_implicit_C_thres', type=int, default=1, help="estimate implicit C_thres based on pol and deltaL")
    parser.add_argument('--negative_event_sampling', type=int, default=0) 
    parser.add_argument('--epoch_start_noEvLoss', type=int, default=0, help="Epoch when to start no-evLoss.")
    parser.add_argument('--weight_loss_rgb', type=float, default=1.0, help="rgb loss weight")
    parser.add_argument('--w_no_ev', type=float, default=1.0, help="rgb loss weight")
    parser.add_argument('--precompute_evs_poses', type=int, default=1, help="preloading poses for each event (much faster, but larger memory required)")
     
    ### training options
    parser.add_argument('--iters', type=int, default=1000000, help="training iters")
    parser.add_argument('--ckpt', type=str, default='latest')
    # parser.add_argument('--lrenc', type=float, default=2e-2) # for event-only: 2e-1
    parser.add_argument('--lr', type=float, default=1e-3, help="initial learning rate") # for event-only: 1e-2
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--num_steps', type=int, default=512, help="num steps sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when not using --cuda_ray)")
    parser.add_argument('--eval_stereo_views', type=int, default=0)
    parser.add_argument('--pp_poses_sphere', type=int, default=1, help="preprocess poses to look at center of sphere")
    parser.add_argument('--render_mode', type=int, default=1, help="Rendering only")

    ### network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")

    ### dataset options
    # (the default value is for the fox dataset)
    parser.add_argument('--mode', type=str, default='eds', help="dataset mode, supports (tumvie, eds, esim)")
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', action='store_true', help="preload all data into GPU, accelerate training but use more GPU memory")
    # (default is for the fox dataset)
    parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=0.33, help="scale adjusts the camera locaction to make sure it falls inside the above bounding box.")
    parser.add_argument('--downscale', type=int, default=1, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--dt_gamma', type=float, default=0, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.1, help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=0.01, help="threshold for density grid to be occupied")
    parser.add_argument('--density_scale', type=float, default=1)   
    parser.add_argument('--bg_radius', type=float, default=-1, help="if positive, use a background model at sphere(bg_radius)")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=1920, help="GUI width")
    parser.add_argument('--H', type=int, default=1080, help="GUI height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=64, help="GUI rendering max sample per pixel")

    ### experimental
    parser.add_argument('--error_map', action='store_true', help="use error map to sample rays")
    parser.add_argument('--clip_text', type=str, default='', help="text input for CLIP guidance")
    parser.add_argument('--rand_poses', type=int, default=0, help="<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses")
    parser.add_argument(                               
        "--infile", help="Pose quat list with desired poses", type=str,
        default="/EXPDIR/val_final_quatlist_kfs_ns.txt"
    )    
    parser.add_argument(                       
        "--N_rand_poses", help="Number of random poses", default=10, type=int
    )

    opt = parser.parse_args()
    return opt

def get_frames(opt):    
    tridxs = opt.train_idxs
    vidxs = opt.val_idxs
    teidxs = opt.test_idxs

    if tridxs is None:
        tridxs = np.arange(2180, 2480, 6).tolist() 
        tridxs = np.arange(3050, 3330, 8).tolist() 
    if vidxs is None:
        vidxs = [2181, 2301, 2401] 
        vidxs = [3091, 3156, 3252]
    if teidxs is None:
        teidxs = [0]
    
    select_frames = {}
    select_frames["train_idxs"] = tridxs
    select_frames["val_idxs"] = vidxs 
    select_frames["test_idxs"] = teidxs 
    
    assert np.all(np.diff(select_frames["train_idxs"]) > 0)
    assert np.all(np.diff(select_frames["val_idxs"]) > 0)
    assert np.all(np.diff(select_frames["test_idxs"]) > 0)
    print(f"Train: {select_frames['train_idxs']}, val: {select_frames['val_idxs']}, test: {select_frames['test_idxs']}")
    assert len(np.unique(select_frames["train_idxs"])) == len(select_frames["train_idxs"])
    assert len(np.unique(select_frames["val_idxs"])) == len(select_frames["val_idxs"])
    assert len(np.unique(select_frames["test_idxs"])) == len(select_frames["test_idxs"])
    return select_frames

def read_all_poses_json(args, opt, file):
    with open(file, 'r') as f:
        transform = json.load(f)

    frames = transform["frames"]
    frames = sorted(frames, key=lambda x: x["file_path"])

    poses = []

    for f in frames:
        f_path = os.path.join(f['file_path'])
        
        pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
        poses.append(pose)

    poses = np.asarray(poses)
    return poses

def interpol_traj_between_rand_poses(poses_train, N_val_poses):
    assert poses_train.shape[0] > 1 and poses_train.shape[1] == 4 and poses_train.shape[2] == 4

    # chose 2 random poses from training set and interpolate in between
    idx0, idx1 = np.random.choice(len(poses_train), 2, replace=False)

    print(f"Selecting val poses between train-pose {idx0} and train-pose {idx1}")
    pose0 = poses_train[idx0] 
    pose1 = poses_train[idx1]
    rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
    slerp = Slerp([0, 1], rots)

    poses_val = np.zeros((N_val_poses + 1, 4, 4))
    for i in range(N_val_poses + 1):
        ratio = np.sin(((i / N_val_poses) - 0.5) * np.pi) * 0.5 + 0.5
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = slerp(ratio).as_matrix()
        pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
        poses_val[i, ...] = pose

    return poses_val, idx0, idx1

def read_calibdata_from_datadir(datadir):
    with open(os.path.join(datadir, "calib_undist.json"), 'r') as f:
        calibdata = json.load(f)["value0"]
    with open(os.path.join(datadir, "mocap-imu-calib.json"), 'r') as f:
        calibdata.update(json.load(f)["value0"])

    return calibdata

def read_calibdata_eds(opt, calibstr="calib0"):
    with open(os.path.join(opt.datadir, f"calib_undist_{calibstr}.json"), 'r') as f:
        calibdata = json.load(f)
        
    return calibdata

def render_path_epi(c2w, up, rads, N):
    render_poses = []
    hwf = c2w[:, 4:5]

    for theta in np.linspace(-1, 1, N + 1)[:-1]:
        # view direction
        c = np.dot(c2w[:3, :4], np.array([theta, 0, 0, 1.]) * rads)
        # camera poses
        z = normalize(np.dot(c2w[:3, :4], np.array([0, 0, 1, 0.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses

def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:, 4:5]

    for theta in np.linspace(0., 2. * np.pi * rots, N + 1)[:-1]:
        # view direction
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads)
        # camera poses
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses

def compute_render_poses(poses, mind=0.5, maxd=2, rad_scale=1):
    c2w = poses_avg(poses)
    print('recentered', c2w.shape)
    print(c2w[:3, :4])

    render_focuspoint_scale = 1  # TODO: tune
    render_radius_scale = rad_scale
    path_epi = False

    ## Get spiral
    up = normalize(poses[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    close_depth, inf_depth = mind * .9, maxd * 5.
    dt = .75
    mean_dz = 1. / (((1. - dt) / close_depth + dt / inf_depth))
    focal = mean_dz
    focal = focal * render_focuspoint_scale
    # Get radii for spiral path
    shrink_factor = .8
    zdelta = close_depth * .2
    tt = poses[:, :3, 3]  
    rads = np.percentile(np.abs(tt), 90, 0)
    rads[0] *= render_radius_scale
    rads[1] *= render_radius_scale
    c2w_path = c2w
    N_views = 120
    N_rots = 2
    # Generate poses for spiral path
    render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
    
    if path_epi:
        rads[0] = rads[0] / 2
        render_poses = render_path_epi(c2w_path, up, rads[0], N_views)

    render_poses = np.array(render_poses).astype(np.float32)

    return render_poses

def update_poses(ps, x=-1.5, y=-0.5, z=-0.75):
    ps[:, 0, 3] += x
    ps[:, 1, 3] += y
    ps[:, 2, 3] += z

    return ps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(                       
        "--model_dir", help="Directory to trained model. Must include transforms.jsonto render randomly around training trajectory", 
        type=str,
        default="EXPDIR"
    )
    parser.add_argument(                               
        "--infile", help="Pose quat list with desired poses", type=str,
        default="EXPDIR/val_final_quatlist_kfs_ns.txt"
    )
    parser.add_argument(                       
        "--rand_poses", help="Boolean: True if render random path, false to provide infile", default=0, type=int
    )
    parser.add_argument(                       
        "--N_rand_poses", help="Number of random poses", default=10, type=int
    )
    parser.add_argument(                       
        "--outdir", help="Directory to trained model. Must include transforms.jsonto render randomly around training trajectory", 
        default=""
    )
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckptdir = os.path.join(args.model_dir, "checkpoints")
    cfgfile = sorted(glob.glob(os.path.join(args.model_dir, "config*.txt")))[-1]
    print(f"cfgfile = {cfgfile}")
    assert os.path.exists(ckptdir)
    assert os.path.isfile(cfgfile)
    outdir = args.outdir

    opt = parse_cfg_file(cfgfile)
    opt.model_dir = args.model_dir
    assert opt.render_mode == 1
    opt.max_ray_batch = 5096 # set low to avoid oom
    model, _, _ = get_model(opt)

    if args.rand_poses == 0: # provide by text file
        assert os.path.isfile(args.infile)
        quatlist = np.loadtxt(args.infile)
        assert quatlist.shape[0] > 0
        assert quatlist.shape[1] == 8
        _, poses_val = quatList_to_poses_hom_and_tss(quatlist)
        outdir = os.path.join(args.model_dir, f"renderFile_{len(poses_val)}")

        #############################################################
        # [tumvie]: Uncomment to eval in right view of tumvie dataset
        # T_imu_camRGBLeft =    {
        #         "px": 0.009108656374744875,
        #         "py": 0.05348659539336455,
        #         "pz": -0.022135445875025417,
        #         "qx": -0.7070921857666745,
        #         "qy": 0.707103565032525,
        #         "qz": -0.0011833937558227545,
        #         "qw": -0.0048773686777871484
        # }

        # T_imu_camRGBRight =    {
        #         "px": 0.00874564726070066,
        #         "py": -0.05599887783697534,
        #         "pz": -0.02160735056816825,
        #         "qx": -0.7071688623418677,
        #         "qy": 0.7070351999059902,
        #         "qz": 0.0025325844762848072,
        #         "qw": -0.0026480641751713287
        # }
        # T_imu_camRGBLeft = quat_dict_to_pose_hom(T_imu_camRGBLeft)
        # T_imu_camRGBRight = quat_dict_to_pose_hom(T_imu_camRGBRight)
        # T_leftRGB_rightRGB = np.linalg.inv(T_imu_camRGBLeft) @ T_imu_camRGBRight
        # poses_val = np.array([poses_val[i, :, :] @ T_leftRGB_rightRGB for i in range(len(poses_val))])[:, 0, :, :]
        
        select_frames = opt.val_idxs
    else:
        tpfile = glob.glob(os.path.join(args.model_dir, "transform*.json"))
        if len(tpfile) == 0:
            tpfile = glob.glob(os.path.join(opt.datadir, "transform*.json"))
        tpfile = tpfile[-1]

        if not os.path.isfile(tpfile):
            # load 'test' = train + val poses
            select_frames = get_frames(opt)
            if opt["events"] == 1:
                test_dl = EventNeRFDataset(args, device=device, type='test', downscale=args.downscale, select_frames=select_frames).dataloader()
            else:
                test_dl = NeRFDataset(args, device=device, type='test', select_frames=select_frames).dataloader()
            poses_train = test_dl.poses
        else: 
            poses_all = read_all_poses_json(args, opt, tpfile)[:, :3, :4] # (N, 3, 4)
            poses_all = np.asarray([nerf_matrix_to_ngp(p, opt.scale) for p in poses_all])
            
            # [debug]: Optional uncomment to plot poses
            # ps = poses_all[:, :3, :].copy()
            # ps[:, :3, 3] *= 50
            # plot_poses(ps, l=10, title="all")
            
            select_frames = get_frames(opt)
            poses_train = poses_all[select_frames["val_idxs"]][:, :3, :4] # chose train_idxs, or val_idxs here

        # [alternative]: computing interpolated poses between training poses
        # poses_train_hom = get_hom_trafos(poses_train[:, :3, :3], poses_train[:, :3, 3])
        # poses_val, _, _ = interpol_traj_between_rand_poses(poses_train_hom, args.N_rand_poses)
        poses_val = compute_render_poses(poses_train, mind=0.9, maxd=1.2, rad_scale=0.2) 
        poses_val = get_hom_trafos(poses_val[:, :3, :3], poses_val[:, :3, 3])[:, :3, :4]

        if len(outdir) == 0:
            outdir = os.path.join(args.model_dir, f"renderRandom_{args.N_rand_poses}")

    if "11_all_characters/" in opt.datadir:
        update_poses(poses_val, x=-1.5, y=-0.5, z=-0.75)
    elif "00_peanuts_dark/" in opt.datadir:
        if not opt.pp_poses_sphere:
            update_poses(poses_val, x=-1, y=-0.5, z=-1)
    elif "ShakeCarpet1" in opt.datadir:
        update_poses(poses_val, x=0, y=0, z=0.3)

    print(f"Plotting val poses:")
    visualize_poses(poses_val, bound=opt.bound)
    
    # Optional uncomment to plot poses_val
    # ps = poses_val[:, :3, :].copy()
    # ps[:, :3, 3] *= 50
    # plot_poses(ps, l=10, title="val")

    # Optional uncomment to plot poses_train&poses_val jointly
    # ps = np.concatenate((poses_train[:, :3, :], poses_val[:, :3, :]))
    # visualize_poses(ps, bound=opt.bound)
    # ps[:, :3, 3] *= 50
    # plot_poses(ps, l=10, title="train & val")

    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "depth"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "raws"), exist_ok=True)
    print(f"Wrting rendered images/depth to {outdir}")

    datadir = opt.datadir
    if opt.mode == "tumvie":
        rendCamId = 0 # chose arbitrary cam. 0/1 = left/right grayscale, 2/3 = left/right event cam
        calibdata = read_calibdata_from_datadir(datadir)
        intrinsics = calibdata["intrinsics_undistorted"][rendCamId]
        intrinsics = [intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]]
        res = calibdata["resolution"][rendCamId]
        W, H = res[0], res[1]
    elif opt.mode == "eds":
        rendCamId = 0 
        calibdata = read_calibdata_eds(opt, calibstr="calib0")
        intrinsics = calibdata["intrinsics_undistorted"][rendCamId]
        intrinsics = [intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]]
        H, W = 480, 640
    else:
        poses_c2w, _ = read_poses_bounds(os.path.join(opt.datadir, "poses_bounds.npy"), invert=False)
        hwf = poses_c2w[0, :3, -1]
        H, W, focal = hwf
        cx, cy = W / 2., H / 2.
        intrinsics = np.array([focal, focal, cx, cy])
    H, W = int(H), int(W)

    criterion = torch.nn.MSELoss(reduction='none')
    trainer = Trainer(opt.expname, opt, model, device=device, criterion=criterion, ema_decay=0.95, fp16=opt.fp16, metrics=[PSNRMeter(opt, select_frames)], use_checkpoint=opt.ckpt)
    model = trainer.model
    poses_val = torch.from_numpy(np.stack(poses_val[:, :3, :4], axis=0).astype(np.float32)).to(device)

    with torch.no_grad():
        for i, pval in enumerate(poses_val):
            rays = get_rays(pval.unsqueeze(0), intrinsics, H, W, -1)
            rays_o, rays_d = rays['rays_o'], rays['rays_d']

            bg_color = 1
            outputs = model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=False, **vars(opt))

            rgb = outputs["image"].detach().cpu().numpy().reshape((H, W, opt.out_dim_color))
            depth = outputs["depth"].detach().cpu().numpy().reshape((H, W, opt.out_dim_color))

            rgbpath = os.path.join(outdir, "rgb", f"{i}.png")
            if opt.mode == "eds" and opt.event_only:
                rgb = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb))
            cv2.imwrite(rgbpath, (rgb * 255).astype(np.uint8))
        
            dpath = os.path.join(outdir,  "depth", f"{i}_depth.png")
            cv2.imwrite(dpath, (depth * 255).astype(np.uint8))

            rawpath = os.path.join(outdir, "raws", f"{i}.png")
            np.save(rawpath, rgb)

            print(f"rendered image {i}/{len(poses_val)}")        

if __name__ == "__main__":
    main()
