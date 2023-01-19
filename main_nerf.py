from re import A
import torch
import configargparse

from nerf.provider import NeRFDataset, EventNeRFDataset
from nerf.gui import NeRFGUI
from nerf.utils import *

from functools import partial
from loss import huber_loss
# [debug] enable for debugging (slow!)
# torch.autograd.set_detect_anomaly(True)

def get_frames(opt):
    tridxs = opt.train_idxs
    vidxs = opt.val_idxs
    teidxs = opt.test_idxs
    eeidxs = opt.exclude_idxs

    if tridxs is None:
        tridxs = np.arange(2850, 3322, 1).tolist() 
        tridxs = np.arange(5, 970, 1).tolist() 
    if vidxs is None:
        vidxs = [2181, 2301, 2401] 
        vidxs = [3091, 3156, 3252] 
    if teidxs is None:
        teidxs = [0]
    
    select_frames = {}
    select_frames["train_idxs"] = tridxs
    select_frames["val_idxs"] = vidxs 
    select_frames["test_idxs"] = teidxs 
    select_frames["exclude_idxs"] = eeidxs
    
    assert np.all(np.diff(select_frames["train_idxs"]) > 0)
    assert np.all(np.diff(select_frames["val_idxs"]) > 0)
    assert np.all(np.diff(select_frames["test_idxs"]) > 0)
    print(f"Train: {select_frames['train_idxs']}, val: {select_frames['val_idxs']}, test: {select_frames['test_idxs']}")
    assert len(np.unique(select_frames["train_idxs"])) == len(select_frames["train_idxs"])
    assert len(np.unique(select_frames["val_idxs"])) == len(select_frames["val_idxs"])
    assert len(np.unique(select_frames["test_idxs"])) == len(select_frames["test_idxs"])
    return select_frames


def get_model(opt): 
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


def assert_config(opt):
    assert opt.acc_max_num_evs >= 0

    if opt.mode == "eds":
        assert opt.pp_poses_sphere == 0

    assert (opt.lr > 1e-7) and (opt.lr < 1e2)
    if opt.event_only:
        assert opt.events == True

    if opt.mode != "tumvie" and opt.mode != "eds":
        assert opt.eval_stereo_views == 0

    if opt.out_dim_color == 1:
        assert opt.use_luma == 0
    assert opt.out_dim_color == 1 or opt.out_dim_color == 3


if __name__ == '__main__':
    parser = configargparse.ArgumentParser() 
    # Dataset and Logging Options
    parser.add_argument(
        "--config",                                      
        default="CONFIGDIR/configs/mocapDesk2/mocapDesk2_nerf.txt",
        is_config_file=True, 
        help="config file path",
    )
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --preload")
    parser.add_argument('--outdir', type=str, default="OUTDIR")
    parser.add_argument('--expweek', type=str, default="testweek")
    parser.add_argument('--expname', type=str, default="testname")
    parser.add_argument('--datadir', type=str, default="DATADIR")  
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
    parser.add_argument('--acc_max_num_evs', type=int, default=0, help="max num successors for event accumulation. if 0: use all, if > 0: use up to max_num (randomly)")
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
    parser.add_argument('--lr', type=float, default=1e-3, help="initial learning rate") 
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--num_steps', type=int, default=512, help="num steps sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when not using --cuda_ray)")
    parser.add_argument('--eval_stereo_views', type=int, default=0)
    parser.add_argument('--pp_poses_sphere', type=int, default=1, help="preprocess poses to look at center of sphere")
    parser.add_argument('--render_mode', type=int, default=0, help="Rendering only")

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
    parser.add_argument('--min_near', type=float, default=0.2, help="minimum near distance for camera")
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
    parser.add_argument('--rand_pose', type=int, default=-1, help="<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses")

    opt = parser.parse_args()
    assert_config(opt)

    model, model_params, encoding_params = get_model(opt)
    select_frames = get_frames(opt)

    criterion = torch.nn.MSELoss(reduction='none')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if opt.test:
        trainer = Trainer(opt.expname, opt, model, device=device, criterion=criterion, fp16=opt.fp16, metrics=[PSNRMeter(opt, select_frames)], use_checkpoint=opt.ckpt)

        if opt.gui:
            gui = NeRFGUI(opt, trainer)
            gui.render()
        
        else:
            test_loader = NeRFDataset(opt, device=device, type='test', select_frames=select_frames).dataloader()
            if opt.mode == 'blender':
                trainer.evaluate(test_loader)
            else:
                trainer.test(test_loader)
            trainer.save_mesh(resolution=256, threshold=10)
    
    else:
        print(f"opt.lr = {opt.lr}")
        optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

        trainer = Trainer(opt.expname, opt, model, device=device, optimizer=optimizer, criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, scheduler_update_every_step=True, metrics=[PSNRMeter(opt, select_frames)], use_checkpoint=opt.ckpt)

        # need different dataset type for GUI/CMD mode.
        if opt.gui:
            train_loader = NeRFDataset(opt, device=device, type='train', select_frames=select_frames).dataloader()
            trainer.train_loader = train_loader # attach dataloader to trainer

            gui = NeRFGUI(opt, trainer)
            gui.render()
        else:
            if opt.events:
                train_loader = EventNeRFDataset(opt, device=device, type='train', downscale=opt.downscale, select_frames=select_frames).dataloader()
                valid_loader = NeRFDataset(opt, device=device, type='val', downscale=opt.downscale, select_frames=select_frames).dataloader()
            else:
                train_loader = NeRFDataset(opt, device=device, type='train', downscale=opt.downscale, select_frames=select_frames).dataloader()
                valid_loader = NeRFDataset(opt, device=device, type='val', downscale=opt.downscale, select_frames=select_frames).dataloader()
            max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
            print(f"max expochs = {max_epoch}")
            trainer.train(train_loader, valid_loader, max_epoch)

            # also test
            test_loader = NeRFDataset(opt, device=device, type='test', select_frames=select_frames).dataloader()       
            trainer.test(test_loader) # colmap doesn't have gt, so just test.
    
            trainer.save_mesh(resolution=256, threshold=10)