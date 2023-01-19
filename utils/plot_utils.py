import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from utils.pose_utils import *

import pyvista as pv
import trimesh
import torch

def render_ev_accumulation(x: np.ndarray, y: np.ndarray, pol: np.ndarray, H: int, W: int) -> np.ndarray:
    assert x.size == y.size == pol.size
    assert H > 0
    assert W > 0
    img = np.full((H,W,3), fill_value=255,dtype='uint8')
    mask = np.zeros((H,W),dtype='int32')
    pol = pol.astype('int')
    pol[pol==0]=-1
    mask1 = (x>=0)&(y>=0)&(W>x)&(H>y)
    mask[y[mask1].astype(np.int), x[mask1].astype(np.int)] = pol[mask1]
    img[mask==0]=[255,255,255]
    img[mask==-1]=[255,0,0]
    img[mask==1]=[0,0,255] 
    return img

def visualize_poses(poses, size=0.1, bound=4):
    # poses: [B, 4, 4]
    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    bound = trimesh.primitives.Box(extents=(bound, bound, bound)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box, bound]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a], [pos, o]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()

def plot_poses(ps_in, path=None, l=0.1, title=""):
    assert ps_in.shape[0] > 0
    assert ps_in.shape[1] == 3 and ps_in.shape[2] == 4
    
    look_vec = np.array([0.0, 0.0, 0.0, 1.0])
    look_vec_x = np.array([1.0, 0.0, 0.0, 1.0])
    look_vec_y = np.array([0.0, 1.0, 0.0, 1.0])
    look_vec_z = np.array([0.0, 0.0, 1.0, 1.0])

    lv = np.squeeze(np.matmul(ps_in, look_vec[:,None]))
    lvx = np.squeeze(np.matmul(ps_in, look_vec_x[:,None]))
    lvy = np.squeeze(np.matmul(ps_in, look_vec_y[:,None]))
    lvz = np.squeeze(np.matmul(ps_in, look_vec_z[:,None]))
    dx = lvx-lv
    dy = lvy-lv
    dz = lvz-lv

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(*lv.T, 'ok')
    if l > 0:
        ax.quiver(*lv.T, *(l*dx.T), color='r', arrow_length_ratio=0.00, label='x')
        ax.quiver(*lv.T, *(l*dy.T), color='g', arrow_length_ratio=0.00, label='y')
        ax.quiver(*lv.T, *(l*dz.T), color='b', arrow_length_ratio=0.00, label='z')
    ax.set_xlim(lvx.min()*2,lvx.max()*2)
    ax.set_ylim(lvy.min()*2,lvx.max()*2)
    ax.set_zlim(lvz.min()*2, lvz.max()*2)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    if l > 0:
        plt.legend()
    plt.title(title)
    plt.show()
    if path is not None:
        plt.savefig(path)

def plot_coord_sys_c2w(
    poses_c2w, ax=None, c2w=True, plot_every_nth_pose=10, text="", title=None, right_handed=True, plot_kf=False
):
    # Input
    # :poses np.array (num_poses, 3, 4) where (3, 4) is (R, t) as cam2world (if c2w==True)

    if len(poses_c2w.shape) == 2:
        poses_c2w = poses_c2w[np.newaxis, ...]
    assert len(poses_c2w.shape) == 3

    poses = np.zeros_like(poses_c2w)
    for i in range(poses_c2w.shape[0]):
        rot = poses_c2w[i, :3, :3]
        trans = poses_c2w[i, :3, 3]
        check_rot(rot, right_handed=right_handed)

        if c2w == False:
            rot, trans = invert_trafo(rot, trans)

        poses[i, :3, :3] = rot
        poses[i, :3, 3] = trans

    ####### Plotting #######
    new_plot = False
    if ax is None:
        new_plot = True
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    if plot_kf:
        color = ["r", "g", "b"]
        label = ["x kf", "y kf", "z kf"]
    else:
        color = ["c", "m", "y"]
        label = ["x all", "y all", "z all"]
    num_poses = poses.shape[0]

    # Positions
    pos = np.zeros((num_poses, 3))
    pos[:, 0] = np.asarray(poses[:, 0, 3])
    pos[:, 1] = np.asarray(poses[:, 1, 3])
    pos[:, 2] = np.asarray(poses[:, 2, 3])

    # Orientations
    # orientation is given as transposed rotation matrices
    x_ori = []
    y_ori = []
    z_ori = []
    for k in range(num_poses):
        x_ori.append(poses[k, :, 0])
        y_ori.append(poses[k, :, 1])
        z_ori.append(poses[k, :, 2])

    x_mod = 1
    y_mod = 1
    z_mod = 1
    # multiply by 10 to visualize orientation more clearly
    pos = pos * [x_mod, y_mod, z_mod]
    dir_vec_x = pos[0] + x_mod * x_ori[0]
    dir_vec_y = pos[0] + y_mod * y_ori[0]
    dir_vec_z = pos[0] + z_mod * z_ori[0]

    if title is not None:
        ax.plot(
            [pos[0, 0], dir_vec_x[0]],
            [pos[0, 1], dir_vec_x[1]],
            [pos[0, 2], dir_vec_x[2]],
            color=color[0],
            label=label[0],
        )
        ax.plot(
            [pos[0, 0], dir_vec_y[0]],
            [pos[0, 1], dir_vec_y[1]],
            [pos[0, 2], dir_vec_y[2]],
            color=color[1],
            label=label[1],
        )
        ax.plot(
            [pos[0, 0], dir_vec_z[0]],
            [pos[0, 1], dir_vec_z[1]],
            [pos[0, 2], dir_vec_z[2]],
            color=color[2],
            label=label[2],
        )
    else:
        ax.plot([pos[0, 0], dir_vec_x[0]], [pos[0, 1], dir_vec_x[1]], [pos[0, 2], dir_vec_x[2]], color=color[0])
        ax.plot([pos[0, 0], dir_vec_y[0]], [pos[0, 1], dir_vec_y[1]], [pos[0, 2], dir_vec_y[2]], color=color[1])
        ax.plot([pos[0, 0], dir_vec_z[0]], [pos[0, 1], dir_vec_z[1]], [pos[0, 2], dir_vec_z[2]], color=color[2])
    ax.text(pos[0, 0], pos[0, 1], pos[0, 2], text)

    label_every_nth_pose = 2
    # if num_poses / plot_every_nth_pose > 500:
    #   plot_every_nth_pose = int(num_poses / 500)
    #   print(plot_every_nth_pose)
    for k in range(1, num_poses):
        if k % plot_every_nth_pose != 0:
            continue
        dir_vec_x = pos[k] + x_mod * x_ori[k]
        dir_vec_y = pos[k] + y_mod * y_ori[k]
        dir_vec_z = pos[k] + z_mod * z_ori[k]
        ax.plot([pos[k, 0], dir_vec_x[0]], [pos[k, 1], dir_vec_x[1]], [pos[k, 2], dir_vec_x[2]], color=color[0])
        ax.plot([pos[k, 0], dir_vec_y[0]], [pos[k, 1], dir_vec_y[1]], [pos[k, 2], dir_vec_y[2]], color=color[1])
        ax.plot([pos[k, 0], dir_vec_z[0]], [pos[k, 1], dir_vec_z[1]], [pos[k, 2], dir_vec_z[2]], color=color[2])
        if k % label_every_nth_pose == 0:
            ax.text(pos[k, 0], pos[k, 1], pos[k, 2], str(k))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    min = np.min([np.min(pos[:, 1]), np.min(pos[:, 0]), np.min(pos[:, 2])])
    max = np.max([np.max(pos[:, 1]), np.max(pos[:, 0]), np.max(pos[:, 2])])
    ax.set_xlim([min, max])
    ax.set_ylim([min, max])
    ax.set_zlim([min, max])
    # plt.gca().invert_yaxis()

    if title is not None:
        if new_plot:
            plt.legend()
            plt.title(title)
            plt.show()
        else:
            plt.legend()
            plt.title(title)
    else:
        plt.legend()
    return ax


def visualize_rays_frames(rays_d, poses, img_i, target_disp, start_frame, end_frame):
    """ "
    Input:
    poses: (num_frames, 3, 4) c2w
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    length_frustum = 2
    scale_trans = 1
    scale_rays = 1
    legend_dict = {"y": "green", "x": "red", "z": "blue", "event ray": "orange", "frame ray": "pink"}
    patchList = []
    for key in legend_dict:
        data_key = mpatches.Patch(color=legend_dict[key], label=key)
        patchList.append(data_key)

    # copy vars
    Poses = np.asarray(poses)
    Poses[:, :, -1] *= scale_trans

    ### PLOT FRAME FRUSTUMS
    plot_coord_sys_c2w(Poses[img_i, :3, :4], ax, length_frustum=length_frustum, text="center", plot_kf=True)
    plot_coord_sys_c2w(Poses[max(img_i - 1, 0), :3, :4], ax, length_frustum=length_frustum, text="prev", plot_kf=True)
    plot_coord_sys_c2w(
        Poses[min(img_i + 1, int(Poses.shape[0]) - 1), :3, :4],
        ax,
        length_frustum=length_frustum,
        text="next",
        plot_kf=True,
    )

    plot_coord_sys_c2w(
        Poses[start_frame, :3, :4], ax, length_frustum=length_frustum, text="start= " + str(start_frame), plot_kf=True
    )
    plot_coord_sys_c2w(
        Poses[end_frame - 1, :3, :4],
        ax,
        length_frustum=length_frustum,
        text="end= " + str(end_frame),
        title="",
        plot_kf=True,
    )

    ray_o = Poses[img_i][:3, -1]
    ##### PLOT DENSE FRAME RAYS
    for i in range(len(rays_d)):
        # if i == 120:
        #    break
        ray_d = np.asarray(rays_d[i]) * scale_rays / target_disp[i]
        ax.plot(
            [ray_o[0], ray_o[0] + ray_d[0]],
            [ray_o[1], ray_o[1] + ray_d[1]],
            [ray_o[2], ray_o[2] + ray_d[2]],
            color="pink",
        )

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()
    plt.legend(handles=patchList)
    return


def plot_sparse_frame_and_evs_rays(
    poses, img_i, frray_at_evs_o, frray_at_evs_d, rays_events_o, rays_events_d, poses_ev, N_rays=30
):
    """ "
    Input:
    poses: (num_frames, 3, 4) c2w
    poses_ev: (N_rand_events, 3, 4) c2w
    """
    # mpl.use("TkAgg")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    scale_rays = 3
    scale_trans = 1  # scale for visualization purpose
    legend_dict = {"y": "green", "x": "red", "z": "blue", "event ray": "orange", "frame ray": "pink"}
    patchList = []
    for key in legend_dict:
        data_key = mpatches.Patch(color=legend_dict[key], label=key)
        patchList.append(data_key)

    # copy vars
    Poses = np.asarray(poses)
    Poses[:, :, -1] *= scale_trans
    Rays_events_o = rays_events_o * scale_trans
    Poses_ev = poses_ev
    Poses_ev[:, :, -1] *= scale_trans

    ### PLOT FRAME FRUSTUMS
    plot_coord_sys_c2w(Poses[img_i, :3, :4], ax, text="center", plot_kf=True)
    plot_coord_sys_c2w(Poses[max(img_i - 1, 0), :3, :4], ax, text="prev", plot_kf=True)
    plot_coord_sys_c2w(Poses[min(img_i + 1, int(Poses.shape[0]) - 1), :3, :4], ax, text="next", plot_kf=True)

    ##### PLOT Sparse FRAME RAYS
    for i in range(len(frray_at_evs_d)):
        if i == N_rays:
            break
        ray_o = frray_at_evs_o[i]
        ray_d = frray_at_evs_d[i] * scale_rays

        ax.plot(
            [ray_o[0], ray_o[0] + ray_d[0]],
            [ray_o[1], ray_o[1] + ray_d[1]],
            [ray_o[2], ray_o[2] + ray_d[2]],
            color="pink",
        )
        ax.text(ray_o[0] + ray_d[0] * 0.05, ray_o[1] + ray_d[1] * 0.05, ray_o[2] + ray_d[2] * 0.05, str(i))

    rays_os = np.zeros((len(rays_events_d), 3))
    #### PLOT SPARSE EVENT FRUSTUMS AND RAYS
    for i in range(len(rays_events_d)):
        if i == N_rays:
            break
        ray_o = Rays_events_o[i]
        ray_d = rays_events_d[i] * scale_rays

        ax.plot(
            [ray_o[0], ray_o[0] + ray_d[0]],
            [ray_o[1], ray_o[1] + ray_d[1]],
            [ray_o[2], ray_o[2] + ray_d[2]],
            color="orange",
        )
        ax.text(ray_o[0], ray_o[1], ray_o[2], str(i))

        # plot_coord_sys_c2w(Poses_ev[i, :, :], ax)
        rays_os[i] = ray_o

    plot_coord_sys_c2w(Poses_ev[i, :, :], ax, title="")

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    plt.show()
    # ax.set_xlim([np.mean(rays_os[:,0]) - dx, np.mean(rays_os[:,0]) + dx])
    # ax.set_ylim([np.mean(rays_os[:,1]) - dx, np.mean(rays_os[:,1]) + dx])
    # ax.set_zlim([np.mean(rays_os[:,2]) - dx, np.mean(rays_os[:,2]) + dx])
    plt.legend(handles=patchList)
    return


def plot_coord_systems_c2w(poses, img_i):
    """ "
    Input:
    poses: (num_frames, 3, 4) c2w
    img_i: Pose selection
    """

    assert poses.shape[0] > 0
    assert poses.shape[1] == 3
    assert poses.shape[2] == 4

    mpl.use("TkAgg")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    scale_trans = 1  # scale for visualization purpose

    legend_dict = {"y": "green", "x": "red", "z": "blue"}
    patchList = []
    for key in legend_dict:
        data_key = mpatches.Patch(color=legend_dict[key], label=key)
        patchList.append(data_key)

    # copy vars
    Poses = np.asarray(poses)

    ### PLOT FRAME FRUSTUMS
    for i in img_i:
        plot_coord_sys_c2w(Poses[i, :3, :4], ax, text=str(i), plot_kf=True)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    plt.show()
    plt.legend(handles=patchList)
    return


def plot_ev_and_frame_rays(
    rays_d, poses, img_i, rays_events_o, rays_events_d, poses_ev, N_dense_rays=50, N_sparse_rays=100
):
    """ "
    Input:
    poses: (num_frames, 3, 4) c2w
    poses_ev: (N_rand_events, 3, 4) c2w
    rays_events_o, rays_events_d: (N_rand_events, 3, 4)
    """
    mpl.use("TkAgg")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    scale_rays = 0.3
    scale_trans = 1  # scale for visualization purpose
    legend_dict = {"y": "green", "x": "red", "z": "blue", "event ray": "orange", "frame ray": "pink"}
    patchList = []
    for key in legend_dict:
        data_key = mpatches.Patch(color=legend_dict[key], label=key)
        patchList.append(data_key)

    # copy vars
    Poses = np.asarray(poses)
    Poses[:, :, -1] *= scale_trans
    Rays_events_o = rays_events_o * scale_trans
    Poses_ev = poses_ev
    Poses_ev[:, :, -1] *= scale_trans

    ### PLOT FRAME FRUSTUMS
    plot_coord_sys_c2w(Poses[img_i, :3, :4], ax, text="center", plot_kf=True)
    plot_coord_sys_c2w(Poses[max(img_i - 1, 0), :3, :4], ax, text="prev", plot_kf=True)
    plot_coord_sys_c2w(Poses[min(img_i + 1, int(Poses.shape[0]) - 1), :3, :4], ax, text="next", plot_kf=True)
    
    ray_o = Poses[img_i, :, -1]
    ##### PLOT DENSE FRAME RAYS
    for i in range(len(rays_d)):
        if i == N_dense_rays:
            break
        ray_d = rays_d[i] * scale_rays  # / target_disp[i]
        ax.plot(
            [ray_o[0], ray_o[0] + ray_d[0]],
            [ray_o[1], ray_o[1] + ray_d[1]],
            [ray_o[2], ray_o[2] + ray_d[2]],
            color="pink",
        )

    rays_os = np.zeros((len(rays_events_d), 3))
    #### PLOT SPARSE EVENT FRUSTUMS AND RAYS
    for i in range(len(rays_events_d)):
        if i == N_sparse_rays:
            break
        ray_o = Rays_events_o[i]
        ray_d = rays_events_d[i] * scale_rays  # / disps_ev[i]

        ax.plot(
            [ray_o[0], ray_o[0] + ray_d[0]],
            [ray_o[1], ray_o[1] + ray_d[1]],
            [ray_o[2], ray_o[2] + ray_d[2]],
            color="orange",
        )
        rays_os[i] = ray_o

    plot_coord_sys_c2w(Poses_ev[i, :, :], ax, title="")

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlim([np.mean(rays_os[:, 0]) - 15, np.mean(rays_os[:, 0]) + 15])
    ax.set_ylim([np.mean(rays_os[:, 1]) - 15, np.mean(rays_os[:, 1]) + 15])
    ax.set_zlim([np.mean(rays_os[:, 2]) - 15, np.mean(rays_os[:, 2]) + 15])
    plt.show()
    return


#########################
# Plotting in provider.py
#########################

def plotting_poses_evs(workspace, rots_hf, trans_hf, tss_hf):
    sorted_idxs = np.argsort(tss_hf)
    tss_hf = tss_hf[sorted_idxs].copy()
    rots_hf = rots_hf[sorted_idxs].copy()
    trans_hf = trans_hf[sorted_idxs].copy()
    
    # idxs = np.asarray(sorted(np.random.rand((2)) * (len(rots_hf)*1/8-1))).astype(np.uint32)
    # if np.diff(idxs) > 5e6:
    #     idxs[0] = idxs[0]
    #     idxs[1] = int(idxs[0] + 5e6)
    # euler_hf = np.asarray([R.from_matrix(r[:3, :3]).as_euler('xyz', degrees=True) for r in rots_hf[idxs[0]:idxs[1]]])
    # tr_hf = np.asarray(trans_hf[idxs[0]:idxs[1]])  #np.asarray([p[:3, 3] for p in ps_hf])

    euler_hf = np.asarray([R.from_matrix(r[:3, :3]).as_euler('xyz', degrees=True) for r in rots_hf[::1000]])
    tr_hf = np.asarray(trans_hf[::1000])  #np.asarray([p[:3, 3] for p in ps_hf])

    # tss_hf = np.asarray([p["ts_ns"]   for p in poses_hf])
    ts_ps = tss_hf[::1000]

    os.makedirs(os.path.join(workspace, "poses"), exist_ok=True)
    n = -1
    plt.figure(figsize=(20,20))
    plt.plot((ts_ps[:n] - ts_ps[0])/1e9, euler_hf[:n, 0], '+-')
    plt.legend(["rot x"])
    plt.title("Rotation event-interpolation")
    plt.xlabel("Time [s]")
    plt.ylabel("Rotation [deg]")
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(workspace, "poses", "evs_rot_x.png"))

    n = -1
    plt.figure(figsize=(20,20))
    plt.plot((ts_ps[:n] - ts_ps[0])/1e9, euler_hf[:n, 1], '+-')
    plt.legend(["rot y"])
    plt.title("Rotation event-interpolation")
    plt.xlabel("Time [s]")
    plt.ylabel("Rotation [deg]")
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(workspace,  "poses", "evs_rot_y.png"))

    n = -1
    plt.figure(figsize=(20,20))
    plt.plot((ts_ps[:n] - ts_ps[0])/1e9, euler_hf[:n, 2], '+-')
    plt.legend(["rot z"])
    plt.title("Rotation event-interpolation")
    plt.xlabel("Time [s]")
    plt.ylabel("Rotation [deg]")
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(workspace, "poses", "evs_rot_z.png"))

    n = -1
    plt.figure(figsize=(20,20))
    plt.plot((ts_ps[:n] - ts_ps[0])/1e9, tr_hf[:n, 0], 'x')
    plt.legend(["trans x"])
    plt.title("Translation event-interpolation")
    plt.xlabel("Time [s]")
    plt.ylabel("Translation [deg]")
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(workspace, "poses", "evs_trans_x.png"))

    n = -1
    plt.figure(figsize=(20,20))
    plt.plot((ts_ps[:n] - ts_ps[0])/1e9, tr_hf[:n, 1], 'x')
    plt.legend(["trans y"])
    plt.title("Translation event-interpolation")
    plt.xlabel("Time [s]")
    plt.ylabel("Translation [deg]")
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(workspace, "poses", "evs_trans_y.png"))

    n = -1
    plt.figure(figsize=(20,20))
    plt.plot((ts_ps[:n] - ts_ps[0])/1e9, tr_hf[:n, 2], 'x')
    plt.legend(["trans z"])
    plt.title("Translation event-interpolation")
    plt.xlabel("Time [s]")
    plt.ylabel("Translation [deg]")
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(workspace, "poses", "evs_trans_z.png"))


def plotting_poses_hf(workspace, poses_hf):
    
    ps_hf = np.asarray([p["pose_c2w"]   for p in poses_hf])
    euler_hf = np.asarray([R.from_matrix(p[:3, :3]).as_euler('xyz', degrees=True) for p in ps_hf])
    tr_hf = np.asarray([p[:3, 3] for p in ps_hf])

    tss_hf = np.asarray([p["ts_ns"]   for p in poses_hf])
    ts_ps = tss_hf-tss_hf[0]
    

    os.makedirs(os.path.join(workspace, "poses"), exist_ok=True)
    n = -int(len(ts_ps)*np.random.rand())
    plt.figure(figsize=(20,20))
    plt.plot((ts_ps[:n] - ts_ps[0])/1e9, euler_hf[:n, 0], '+-')
    plt.legend(["rot x"])
    plt.title("Rotation MoCap-raw")
    plt.xlabel("Time [s]")
    plt.ylabel("Rotation [deg]")
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(workspace, "poses", "mocap_rot_x.png"))

    n = -int(len(ts_ps)*np.random.rand())
    plt.figure(figsize=(20,20))
    plt.plot((ts_ps[:n] - ts_ps[0])/1e9, euler_hf[:n, 1], '+-')
    plt.legend(["rot y"])
    plt.title("Rotation MoCap-raw")
    plt.xlabel("Time [s]")
    plt.ylabel("Rotation [deg]")
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(workspace,  "poses", "mocap_rot_y.png"))

    n = -int(len(ts_ps)*np.random.rand())
    plt.figure(figsize=(20,20))
    plt.plot((ts_ps[:n] - ts_ps[0])/1e9, euler_hf[:n, 2], '+-')
    plt.legend(["rot z"])
    plt.title("Rotation MoCap-raw")
    plt.xlabel("Time [s]")
    plt.ylabel("Rotation [deg]")
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(workspace, "poses", "mocap_rot_z.png"))

    n = -int(len(ts_ps)*np.random.rand())
    plt.figure(figsize=(20,20))
    plt.plot((ts_ps[:n] - ts_ps[0])/1e9, tr_hf[:n, 0], 'x')
    plt.legend(["trans x"])
    plt.title("Translation MoCap-raw")
    plt.xlabel("Time [s]")
    plt.ylabel("Translation [deg]")
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(workspace, "poses", "mocap_trans_x.png"))

    n = -int(len(ts_ps)*np.random.rand())
    plt.figure(figsize=(20,20))
    plt.plot((ts_ps[:n] - ts_ps[0])/1e9, tr_hf[:n, 1], 'x')
    plt.legend(["trans y"])
    plt.title("Translation MoCap-raw")
    plt.xlabel("Time [s]")
    plt.ylabel("Translation [deg]")
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(workspace, "poses", "mocap_trans_y.png"))

    n = -int(len(ts_ps)*np.random.rand())
    plt.figure(figsize=(20,20))
    plt.plot((ts_ps[:n] - ts_ps[0])/1e9, tr_hf[:n, 2], 'x')
    plt.legend(["trans z"])
    plt.title("Translation MoCap-raw")
    plt.xlabel("Time [s]")
    plt.ylabel("Translation [deg]")
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(workspace, "poses", "mocap_trans_z.png"))



def plotting_poses_bounds(workspace, tss_imgs_us, poses_bounds):
    ts_ps = tss_imgs_us-tss_imgs_us[0]
    ps_bds = np.asarray(poses_bounds)[:, :-2].reshape([-1, 3, 5])[:, :3, :4]  # (num_poses, 17) to (num_poses, 3, 5)
    euler_hf = np.asarray([R.from_matrix(p[:3, :3]).as_euler('xyz', degrees=True) for p in ps_bds])
    tr_hf = np.asarray([p[:3, 3] for p in ps_bds])

    os.makedirs(os.path.join(workspace, "poses"), exist_ok=True)
    n = -int(len(ts_ps)*np.random.rand())
    plt.figure(figsize=(20,20))
    plt.plot((ts_ps[:n] - ts_ps[0])/1e6, euler_hf[:n, 0], '+-')
    plt.legend(["rot x"])
    plt.title("Rotation x from MoCap")
    plt.xlabel("Time [s]")
    plt.ylabel("Rotation [deg]")
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(workspace, "poses", "interpolFrame_rot_x.png"))

    n = -int(len(ts_ps)*np.random.rand())
    plt.figure(figsize=(20,20))
    plt.plot((ts_ps[:n] - ts_ps[0])/1e6, euler_hf[:n, 1], '+-')
    plt.legend(["rot y"])
    plt.title("Rotation y from MoCap")
    plt.xlabel("Time [s]")
    plt.ylabel("Rotation [deg]")
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(workspace,  "poses", "interpolFrame_rot_y.png"))

    n = -int(len(ts_ps)*np.random.rand())
    plt.figure(figsize=(20,20))
    plt.plot((ts_ps[:n] - ts_ps[0])/1e6, euler_hf[:n, 2], '+-')
    plt.legend(["rot z"])
    plt.title("Rotation z from MoCap")
    plt.xlabel("Time [s]")
    plt.ylabel("Rotation [deg]")
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(workspace, "poses", "interpolFrame_rot_z.png"))

    n = -int(len(ts_ps)*np.random.rand())
    plt.figure(figsize=(20,20))
    plt.plot((ts_ps[:n] - ts_ps[0])/1e6, tr_hf[:n, 0], 'x')
    plt.legend(["trans x"])
    plt.title("Translation x from MoCap")
    plt.xlabel("Time [s]")
    plt.ylabel("Translation [deg]")
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(workspace, "poses", "interpolFrame_trans_x.png"))

    n = -int(len(ts_ps)*np.random.rand())
    plt.figure(figsize=(20,20))
    plt.plot((ts_ps[:n] - ts_ps[0])/1e6, tr_hf[:n, 1], 'x')
    plt.legend(["trans y"])
    plt.title("Translation y from MoCap")
    plt.xlabel("Time [s]")
    plt.ylabel("Translation [deg]")
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(workspace, "poses", "interpolFrame_trans_y.png"))

    n = -int(len(ts_ps)*np.random.rand())
    plt.figure(figsize=(20,20))
    plt.plot((ts_ps[:n] - ts_ps[0])/1e6, tr_hf[:n, 2], 'x')
    plt.legend(["trans z"])
    plt.title("Translation z from MoCap")
    plt.xlabel("Time [s]")
    plt.ylabel("Translation [deg]")
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(workspace, "poses", "interpolFrame_trans_z.png"))