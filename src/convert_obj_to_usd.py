#!/usr/bin/env python3
"""
Pipeline Part 2: Mesh OBJ -> USD (via Isaac Sim headless)

You should already have:
    <scene_dir>/output/<something>.obj

Example:
    C:\isaac-sim\python.bat scripts\convert_obj_to_usd.py `
        --mesh "C:\path\to\bottle02_mesh.obj" `
        --usd  "C:\path\to\bottle02.usd"
"""

from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True})

import argparse
from pathlib import Path

def convert_mesh_to_usd_with_isaac(mesh_obj: Path, usd_out: Path):
    import asyncio
    import carb
    import omni.kit.asset_converter as asset_converter

    def progress_callback(progress, total_steps):
        print(f"[Convert] {progress}/{total_steps}")

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

def main():
    parser = argparse.ArgumentParser(
        description="Part 2: Convert NeRF mesh OBJ to USD using Isaac Sim headless."
    )
    parser.add_argument(
        "--mesh",
        type=str,
        required=True,
        help="Path to the input OBJ mesh (exported from instant-ngp).",
    )
    parser.add_argument(
        "--usd",
        type=str,
        default="",
        help="Output USD path. If omitted, uses same folder/name with .usd extension.",
    )
    args = parser.parse_args()

    mesh_path = Path(args.mesh).resolve()
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh OBJ not found: {mesh_path}")

    if args.usd:
        usd_path = Path(args.usd).resolve()
    else:
        usd_path = mesh_path.with_suffix(".usd")

    print(f"Input mesh OBJ : {mesh_path}")
    print(f"Output USD     : {usd_path}")

    usd_path.parent.mkdir(parents=True, exist_ok=True)

    convert_mesh_to_usd_with_isaac(mesh_path, usd_path)

    if usd_path.exists():
        print("\n" + "=" * 60)
        print("=== USD CONVERSION COMPLETE ===")
        print("=" * 60)
        print(f"USD asset       : {usd_path}")
        print("=" * 60)
    else:
        raise RuntimeError(f"USD file was not created at: {usd_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
