from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni.replicator.core as rep
from pathlib import Path
import os

import shutil
import time

# ---------- Paths / setup ----------
ROOT = Path(__file__).resolve().parents[1]   # project root (up from scripts/)
OUTPUT_DIR = ROOT / "data" / "datasets" / "cube_test"

# Clean out old run completely
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n=== Replicator cube test starting (instrumented) ===")
print(f"cwd             = {os.getcwd()}")
print(f"OUTPUT_DIR      = {OUTPUT_DIR}")
print(f"OUTPUT_DIR abs? = {OUTPUT_DIR.is_absolute()}")
print(f"OUTPUT_DIR exists? {OUTPUT_DIR.exists()}")

# For sanity, show all registered writers
print("Available writers in WriterRegistry:", rep.WriterRegistry.get_writers())

# ---------- Build scene ----------