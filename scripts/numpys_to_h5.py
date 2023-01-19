import numpy as np
import os
import h5py
import argparse

import glob
from natsort import natsorted
from tqdm import tqdm
from utils.event_utils import compute_ms_to_idx

def render(x, y, pol, H, W):
    assert x.size == y.size == pol.size
    assert H > 0
    assert W > 0
    img = np.full((H,W,3), fill_value=255,dtype='uint8')
    mask = np.zeros((H,W),dtype='int32')
    pol = pol.astype('int')
    pol[pol==0]=-1
    mask1 = (x>=0)&(y>=0)&(W>x)&(H>y)
    mask[y[mask1],x[mask1]]=pol[mask1]
    img[mask==0]=[255,255,255]
    img[mask==-1]=[255,0,0]
    img[mask==1]=[0,0,255]
    return img


def main():
    # Reading my data
    parser = argparse.ArgumentParser(description="Resizes images in dir")
    parser.add_argument(         
        "--indir", help="Input image directory.", default="/DATADIR/events/" 
    )
    args = parser.parse_args()
    print(f"{args.indir}")

    numpy_lists = natsorted(glob.glob(os.path.join(args.indir ,"*.npy")))
    xs, ys, ps, ts = [], [], [], []
    pbar = tqdm(total=len(numpy_lists))
    for num in numpy_lists:
        tmp = np.load(num, allow_pickle=True)
        xs.append(np.array(tmp[:, 0]).astype(np.uint16))
        ys.append(np.array(tmp[:, 1]).astype(np.uint16))
        ps.append(np.array(tmp[:, 3]).astype(np.uint8)) 
        if "ESIM" in args.indir:
            ts.append(np.array(tmp[:, 2]/1000).astype(np.int64)) # ns => us
        else:
            ts.append(np.array(tmp[:, 2]).astype(np.int64))
        pbar.update(1)

    print(f"loaded all data in RAM")    
    xs = np.concatenate(xs) # (N,)
    ys = np.concatenate(ys)
    ps = np.concatenate(ps)
    ts = np.concatenate(ts)

    if ps.max() == 255:
        ps[ps == 255] = 0

    Nfiltered = len(xs)
    efoutpath = os.path.join(os.path.dirname(args.indir), args.indir[-9:], "events.h5") 
    os.makedirs(os.path.dirname(efoutpath), exist_ok=True)
    print(f"saving to {efoutpath}")
    ef_out = h5py.File(efoutpath, 'w')
    ef_out.clear()
    ef_out.create_dataset('x', shape=(Nfiltered), dtype=np.uint16, compression="lzf")
    ef_out.create_dataset('y', shape=(Nfiltered), dtype=np.uint16, compression="lzf")
    ef_out.create_dataset('t', shape=(Nfiltered), dtype=np.int64, compression="lzf")
    ef_out.create_dataset('p', shape=(Nfiltered), dtype=np.uint8, compression="lzf")
    ef_out["x"][:] = xs
    ef_out["y"][:] = ys
    ef_out["t"][:] = ts
    ef_out["p"][:] = ps

    print("Start computing ms to idx")
    ms_to_idx = compute_ms_to_idx(ts*1000)
    ef_out.create_dataset('ms_to_idx', shape=len(ms_to_idx), dtype=np.uint64, compression="lzf")
    ef_out["ms_to_idx"][:] = ms_to_idx
    print(f"Done with ms_to_idx")
    
    ef_out.close()

if __name__ == '__main__':
    main()
