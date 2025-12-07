#!/usr/bin/env python3
"""
Pipeline Part 1: Images -> COLMAP -> transforms.json

1) Copies source images into a per-scene workspace:
       <workspace_root>/<scene_name>/
           images/
           colmap/

2) Runs colmap2nerf_compat.py to:
       - run COLMAP (feature extraction, matching, mapper, bundle adjuster)
       - export COLMAP model to text
       - generate transforms.json for instant-ngp

You then manually:

   - Launch instant-ngp GUI and point it at <workspace_root>/<scene_name>
   - Train the NeRF
   - Export a mesh OBJ into <workspace_root>/<scene_name>/output/
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# ====== USER CONFIG ======

# Where to store per-object workspaces (your project's data/ folder)
DEFAULT_DATA_ROOT = (Path(__file__).resolve().parent.parent / "data").resolve()

# Path to your Real-to-Sim project repo (this repo)
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # adjust if needed

# NeRF / scene scaling for colmap2nerf
AABB_SCALE = 16  # typical for object-scale scenes

# ====== UTILS ======

def run_cmd(cmd, cwd=None):
    print(f"\n[RUN] {cmd}\n   (cwd={cwd})")
    result = subprocess.run(cmd, cwd=cwd, check=True)
    return result


def ensure_empty_dir(path: Path):
    if path.exists():
        # be careful here: this deletes previous results
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

# ====== STEP 1: Prepare workspace & images ======

def prepare_workspace(images_dir: Path, workspace_root: Path, scene_name: str) -> Path:
    """
    Create a workspace for this object and copy images into scene_dir/images.

    Returns:
        scene_dir: <workspace_root>/<scene_name>
    """
    scene_dir = workspace_root / scene_name
    images_out = scene_dir / "images"
    colmap_dir = scene_dir / "colmap"

    scene_dir.mkdir(parents=True, exist_ok=True)
    images_out.mkdir(parents=True, exist_ok=True)
    colmap_dir.mkdir(parents=True, exist_ok=True)

    # Simple implementation: copy images.
    for img in sorted(images_dir.glob("*")):
        if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            dst = images_out / img.name
            if not dst.exists():
                shutil.copy2(img, dst)

    print(f"Prepared workspace at: {scene_dir}")
    print(f"Copied images to:      {images_out}")
    return scene_dir

# ====== STEP 2: Run colmap2nerf_compat (which runs COLMAP) ======
# This estimates camera poses and creates transforms.json for NeRF.

def run_colmap2nerf(scene_dir: Path):
    """
    Calls scripts/third_party/colmap2nerf_compat.py to:
      - run COLMAP SfM
      - produce transforms.json in scene_dir
      - export COLMAP model to text
    """
    images_dir = scene_dir / "images"
    colmap_db = scene_dir / "colmap" / "database.db"
    colmap_sparse = scene_dir / "colmap" / "sparse"
    transforms_path = scene_dir / "transforms.json"

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "third_party" / "colmap2nerf_compat.py"),
        "--run_colmap",
        "--colmap_matcher", "exhaustive",
        "--aabb_scale", str(AABB_SCALE),
        "--images", str(images_dir),
        "--out", str(transforms_path),
        "--colmap_db", str(colmap_db),
        "--text", str(colmap_sparse),
    ]

    # colmap2nerf_compat.py lives in your project repo
    run_cmd(cmd, cwd=PROJECT_ROOT)

    if not transforms_path.exists():
        raise RuntimeError(f"colmap2nerf_compat.py did not produce {transforms_path}")

    print(f"Generated transforms.json at: {transforms_path}")
    return transforms_path

# ====== MAIN ======

def main():
    parser = argparse.ArgumentParser(
        description="Part 1: Images -> COLMAP (colmap2nerf_compat) -> transforms.json"
    )
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Directory of input images for this object (your DSC*.jpgs).",
    )
    parser.add_argument(
        "--scene_name",
        type=str,
        required=True,
        help="Short name for this object (e.g. 'bottle02').",
    )
    parser.add_argument(
        "--workspace_root",
        type=str,
        default=str(DEFAULT_DATA_ROOT),
        help=f"Root folder for per-object workspaces (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--skip_colmap",
        action="store_true",
        help="Skip COLMAP step (requires pre-existing transforms.json in scene_dir).",
    )
    args = parser.parse_args()

    images_dir = Path(args.images).resolve()
    workspace_root = Path(args.workspace_root).resolve()
    scene_name = args.scene_name

    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")

    workspace_root.mkdir(parents=True, exist_ok=True)

    # Step 1: prepare scene workspace
    scene_dir = prepare_workspace(images_dir, workspace_root, scene_name)

    # Step 2: COLMAP + transforms.json
    transforms_path = scene_dir / "transforms.json"
    if not args.skip_colmap:
        print("\n[STEP 2] Running COLMAP + colmap2nerf_compat.py...")
        run_colmap2nerf(scene_dir)

        # Verify COLMAP output
        if transforms_path.exists():
            print(f"✓ transforms.json created at: {transforms_path}")
            print(f"  File size: {transforms_path.stat().st_size} bytes")
        else:
            print(f"✗ ERROR: transforms.json NOT found at: {transforms_path}")
            print(f"  Scene dir contents:")
            for item in sorted(scene_dir.iterdir()):
                if item.is_dir():
                    print(f"    [DIR]  {item.name}/")
                else:
                    print(f"    [FILE] {item.name} ({item.stat().st_size} bytes)")
            raise RuntimeError("COLMAP step failed: transforms.json not produced")
    else:
        print("\n[STEP 2] Skipping COLMAP (--skip_colmap flag set)")
        if transforms_path.exists():
            print(f"✓ Found existing transforms.json at: {transforms_path}")
            print(f"  File size: {transforms_path.stat().st_size} bytes")
        else:
            print(f"✗ ERROR: transforms.json not found and --skip_colmap was set")
            print(f"  Expected at: {transforms_path}")
            raise RuntimeError("COLMAP step skipped but transforms.json not found")

    print("\n" + "=" * 60)
    print("=== PART 1 COMPLETE ===")
    print("=" * 60)
    print(f"Scene directory : {scene_dir}")
    print(f"Transforms JSON : {transforms_path}")
    print("\nNext steps:")
    print(f"  1) Open instant-ngp GUI and run:")
    print(f"       cd C:\\Apps\\Instant-NGP-for-RTX-2000")
    print(f"       .\\instant-ngp.exe --scene \"{scene_dir}\"")
    print("  2) Train NeRF until satisfied.")
    print(f"  3) Export mesh OBJ into: {scene_dir / 'output'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
