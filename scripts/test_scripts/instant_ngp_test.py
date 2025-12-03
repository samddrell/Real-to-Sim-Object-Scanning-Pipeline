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
DEFAULT_WORKSPACE_ROOT = Path(r"C:\nerf_workspaces").resolve()

# NeRF training settings
N_STEPS = 30000  # tune based on quality/time tradeoff
AABB_SCALE = 16  # typical for object-scale scenes

# Marching cubes / mesh fidelity knob is controlled inside run.py via default settings,
# plus mesh resolution flags in some builds. 
