#!/usr/bin/env python3
"""
End-to-end pipeline:

1) images -> COLMAP + transforms.json (via colmap2nerf.py)
2) transforms.json -> trained NeRF + mesh.obj (via scripts/run.py)
3) mesh.obj -> mesh.usd (via headless Isaac)

"""
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True})

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
DEFAULT_WORKSPACE_ROOT = (Path(__file__).resolve().parent.parent.parent / "data").resolve()

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
        "--colmap_db", str(colmap_db),
        "--text", str(colmap_sparse),
    ]

    # colmap2nerf wants to be run from instant-ngp root usually
    run_cmd(cmd, cwd=INSTANT_NGP_ROOT)

    if not transforms_path.exists():
        raise RuntimeError(f"colmap2nerf did not produce {transforms_path}")

    print(f"Generated transforms.json at: {transforms_path}")
    return transforms_path

# ====== STEP 3: Train NeRF + export mesh with run.py ======

def train_nerf_and_export_mesh(scene_dir: Path, mesh_out: Path, snapshot_out: Path):
    """
    Uses scripts/run.py in instant-ngp to:
      - train for N_STEPS iterations
      - save a snapshot
      - export a marching-cubes mesh to mesh_out (OBJ/PLY depending on extension)
    """
    cmd = [
        sys.executable,
        str(INSTANT_NGP_ROOT / "scripts" / "run.py"),
        "--mode", "nerf",
        "--scene", str(scene_dir),
        "--n_steps", str(N_STEPS),
        "--save_snapshot", str(snapshot_out),
        "--save_mesh", str(mesh_out),
    ]

    # This leverages run.py's --save_mesh flag, which triggers marching cubes
    # and writes an OBJ/PLY mesh of the NeRF. :contentReference[oaicite:6]{index=6}
    run_cmd(cmd, cwd=INSTANT_NGP_ROOT)

    if not mesh_out.exists():
        raise RuntimeError(f"run.py did not produce mesh at {mesh_out}")

    print(f"Saved NeRF snapshot to: {snapshot_out}")
    print(f"Exported mesh to:       {mesh_out}")


# ====== STEP 4: ISAAC headless OBJ -> USD ======

def convert_mesh_to_usd_with_isaac(mesh_obj: Path, usd_out: Path):
    import asyncio
    import carb
    import omni.kit.asset_converter as asset_converter

    def progress_callback(progress, total_steps):
        print(f"[Convert] {progress}/{total_steps}")

    # Set up conversion context (flags optional, these are just defaults)
    ctx = asset_converter.AssetConverterContext()

    instance = asset_converter.get_instance()
    task = instance.create_converter_task(
        str(mesh_obj),
        str(usd_out),
        progress_callback,
        ctx,
    )

    loop = asyncio.get_event_loop()
    success = loop.run_until_complete(task.wait_until_finished())

    if not success:
        carb.log_error(task.get_status(), task.get_detailed_error())
        raise RuntimeError("Asset conversion failed")

    print(f"[OK] Converted {mesh_obj} -> {usd_out}")

# ====== MAIN PIPELINE ======

def main():
    parser = argparse.ArgumentParser(
        description="Images -> NeRF (instant-ngp) -> Mesh (OBJ) -> USD (Isaac headless)"
    )
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Directory of input images for this object",
    )
    parser.add_argument(
        "--scene_name",
        type=str,
        required=True,
        help="Short name for this object (e.g. 'mug01')",
    )
    parser.add_argument(
        "--workspace_root",
        type=str,
        default=str(DEFAULT_WORKSPACE_ROOT),
        help=f"Root folder for per-object workspaces (default: {DEFAULT_WORKSPACE_ROOT})",
    )
    parser.add_argument(
        "--skip_colmap",
        action="store_true",
        help="Skip COLMAP step (requires pre-existing transforms.json)",
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
    if not args.skip_colmap:
        print("\n[STEP 2] Running COLMAP + colmap2nerf...")
        run_colmap2nerf(scene_dir)
        
        # Verify COLMAP output
        transforms_path = scene_dir / "transforms.json"
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
        transforms_path = scene_dir / "transforms.json"
        if transforms_path.exists():
            print(f"✓ Found existing transforms.json at: {transforms_path}")
            print(f"  File size: {transforms_path.stat().st_size} bytes")
        else:
            print(f"✗ ERROR: transforms.json not found and --skip_colmap was set")
            print(f"  Expected at: {transforms_path}")
            raise RuntimeError("COLMAP step skipped but transforms.json not found")

    # Step 3: train NeRF + export mesh
    output_dir = scene_dir / "output"
    output_dir.mkdir(exist_ok=True)

    mesh_out = output_dir / f"{scene_name}_mesh.obj"
    snapshot_out = output_dir / f"{scene_name}_nerf.msgpack"

    print("\n[STEP 3] Training NeRF and exporting mesh...")
    train_nerf_and_export_mesh(scene_dir, mesh_out, snapshot_out)
    
    # Verify NeRF training output
    if mesh_out.exists():
        print(f"✓ Mesh OBJ created at: {mesh_out}")
        print(f"  File size: {mesh_out.stat().st_size} bytes")
    else:
        print(f"✗ ERROR: Mesh OBJ NOT found at: {mesh_out}")
        print(f"  Output dir contents:")
        if output_dir.exists():
            for item in sorted(output_dir.iterdir()):
                if item.is_dir():
                    print(f"    [DIR]  {item.name}/")
                else:
                    print(f"    [FILE] {item.name} ({item.stat().st_size} bytes)")
        else:
            print(f"    Output dir does not exist: {output_dir}")
        raise RuntimeError("NeRF training step failed: mesh OBJ not produced")
    
    if snapshot_out.exists():
        print(f"✓ NeRF snapshot created at: {snapshot_out}")
        print(f"  File size: {snapshot_out.stat().st_size} bytes")
    else:
        print(f"⚠ WARNING: NeRF snapshot not found at: {snapshot_out}")
        print(f"  This may be okay if run.py doesn't support --save_snapshot")

    # Step 4: OBJ -> USD via Isaac headless
    usd_out = output_dir / f"{scene_name}.usd"
    temp_script = output_dir / "convert_obj_to_usd_tmp.py"

    print("\n[STEP 4] Converting OBJ mesh to USD via Isaac...")
    try:
        convert_mesh_to_usd_with_isaac(mesh_out, usd_out)
    except Exception as e:
        print(f"✗ ERROR during USD conversion: {e}")
        print(f"  Input mesh: {mesh_out} (exists: {mesh_out.exists()})")
        print(f"  Output path: {usd_out}")
        raise
    
    # Verify USD conversion output
    if usd_out.exists():
        print(f"✓ USD asset created at: {usd_out}")
        print(f"  File size: {usd_out.stat().st_size} bytes")
    else:
        print(f"✗ ERROR: USD asset NOT found at: {usd_out}")
        print(f"  Output dir contents after conversion:")
        for item in sorted(output_dir.iterdir()):
            if item.is_dir():
                print(f"    [DIR]  {item.name}/")
            else:
                print(f"    [FILE] {item.name} ({item.stat().st_size} bytes)")
        raise RuntimeError("USD conversion step failed: USD file not produced")

    print("\n" + "="*60)
    print("=== PIPELINE COMPLETE ===")
    print("="*60)
    print(f"Scene directory : {scene_dir}")
    print(f"Mesh OBJ        : {mesh_out}")
    print(f"USD asset       : {usd_out}")
    print("\nYou can now bring this USD into Isaac Sim / Omniverse Replicator.")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    finally:
        # Cleanly shut down Isaac / Kit
        simulation_app.close()
