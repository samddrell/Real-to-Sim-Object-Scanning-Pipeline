#!/usr/bin/env python3
"""
End-to-end pipeline:

1) images -> COLMAP + transforms.json (via colmap2nerf.py)
2) transforms.json -> trained NeRF + mesh.obj (via scripts/run.py)
3) mesh.obj -> mesh.usd (via headless Blender)

"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ====== USER CONFIG ======

# Path to  instant-ngp repo root (where CMakeLists.txt lives)
INSTANT_NGP_ROOT = Path(r"C:\Apps\Instant-NGP-for-RTX-2000").resolve()

# Path to Isaac Sim python launcher
ISAAC_PYTHON = Path(r"C:\isaac-sim\python.bat").resolve()

# Where to store per-object workspaces
DEFAULT_WORKSPACE_ROOT = Path(__file__).resolve().parent / "data"

# NeRF training settings
N_STEPS = 30000  # tune based on quality/time tradeoff
AABB_SCALE = 16  # typical for object-scale scenes

# Marching cubes / mesh fidelity knob is controlled inside run.py via default settings,
# plus mesh resolution flags in some builds. 

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
    Create a workspace for this object and copy/symlink images into scene_dir/images.
    Returns the path to scene_dir.
    """
    scene_dir = workspace_root / scene_name
    images_out = scene_dir / "images"
    colmap_dir = scene_dir / "colmap"

    scene_dir.mkdir(parents=True, exist_ok=True)
    images_out.mkdir(parents=True, exist_ok=True)
    colmap_dir.mkdir(parents=True, exist_ok=True)

    # Simple implementation: copy images.
    # You could instead symlink to save space.
    for img in sorted(images_dir.glob("*")):
        if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            dst = images_out / img.name
            if not dst.exists():
                shutil.copy2(img, dst)

    print(f"Prepared workspace at: {scene_dir}")
    print(f"Copied images to:      {images_out}")
    return scene_dir

# ====== STEP 2: Run colmap2nerf (which runs COLMAP) ======
# This estimates camera poses and creates transforms.json for nerf training.

def run_colmap2nerf(scene_dir: Path):
    """
    Calls scripts/colmap2nerf.py to:
      - run COLMAP SfM
      - produce transforms.json in scene_dir
      - organize images
    """
    images_dir = scene_dir / "images"
    colmap_db = scene_dir / "colmap" / "database.db"
    colmap_sparse = scene_dir / "colmap" / "sparse"
    transforms_path = scene_dir / "transforms.json"

    cmd = [
        sys.executable,
        str(INSTANT_NGP_ROOT / "scripts" / "colmap2nerf.py"),
        "--run_colmap",
        "--colmap_matcher", "exhaustive",
        "--aabb_scale", str(AABB_SCALE),
        "--images", str(images_dir),
        "--out", str(transforms_path),
        "--db", str(colmap_db),
        "--text", str(colmap_sparse),
    ]

    # colmap2nerf wants to be run from instant-ngp root usually
    run_cmd(cmd, cwd=INSTANT_NGP_ROOT)

    if not transforms_path.exists():
        raise RuntimeError(f"colmap2nerf did not produce {transforms_path}")

    print(f"Generated transforms.json at: {transforms_path}")
    return transforms_path
