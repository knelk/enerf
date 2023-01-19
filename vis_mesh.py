import numpy as np
import trimesh
import glob, os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Resizes images in dir")
    parser.add_argument(       
        "--infile", help="Input image directory.", default="RUNDIR/meshes/"
    )
    args = parser.parse_args()
    # mf = glob.glob(os.path.join(args.indir, "*ply"))[0]

    assert ".ply" in args.infile
    print(f"loading mesh from {args.infile}")
    
    mesh = trimesh.exchange.load.load(args.infile, file_type="ply")
    mesh.show()

if __name__ == "__main__":
    main()
