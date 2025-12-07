from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import os
import argparse

from isaacsim.core.api import World
import omni.replicator.core as rep

# ---------------------------------------------------------------------
# Argument parsing (for USD + basic config)
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Isaac Replicator subject test", add_help=False)
parser.add_argument(
    "--usd",
    type=str,
    required=True,
    help="Path to the subject .usd file (relative or absolute)",
)
parser.add_argument(
    "--frames",
    type=int,
    default=50,
    help="Number of frames to render",
)
parser.add_argument(
    "--out",
    type=str,
    default="output_images_2",
    help="Output directory for rendered images (relative to CWD)",
)

# Isaac / Kit inject a bunch of extra args; ignore them
args, _ = parser.parse_known_args()
usd_path = os.path.abspath(args.usd)
target_frames = args.frames
output_subdir = args.out

print("CWD:", os.getcwd())
print("USD path:", usd_path)
print("Target frames:", target_frames)
print("Output subdir:", output_subdir)

if not os.path.exists(usd_path):
    raise FileNotFoundError(f"Subject USD not found: {usd_path}")

# ---------------------------------------------------------------------
# World + ground
# ---------------------------------------------------------------------
world = World(
    stage_units_in_meters=1.0,
    physics_prim_path="/physicsScene",
    backend="numpy",
)
world.scene.add_default_ground_plane()
world.reset()
world.play()
print("WORLD PLAYING?", world.is_playing())

# ---------------------------------------------------------------------
# Load subject USD
# ---------------------------------------------------------------------
# We treat this as the main object of interest in the scene.
# The returned object is a ReplicatorItem we can use for look_at later.
subject = rep.create.from_usd(
    usd_path,
    semantics=[("class", "subject")],
)
# Optionally adjust its pose (e.g., center on origin, slightly above ground)
with subject:
    rep.modify.pose(
        position=(0.0, 0.0, 0.0),
        # You can tweak this if your asset is off-center or rotated oddly
        rotation=(0.0, 0.0, 0.0),
    )

print("Loaded subject from USD.")

# ---------------------------------------------------------------------
# Camera + render product + PNG writer
# ---------------------------------------------------------------------
# Simple camera looking at the subject
camera = rep.create.camera(
    position=(2.0, 2.0, 2.0),
    look_at=subject,   # we can pass the ReplicatorItem directly
)

rp = rep.create.render_product(
    camera,
    resolution=(640, 480),
)

writer_type = "BasicWriter"
writer = rep.WriterRegistry.get(writer_type)

output_dir = os.path.join(os.getcwd(), output_subdir)
os.makedirs(output_dir, exist_ok=True)

writer.initialize(
    output_dir=output_dir,
    rgb=True,
)
writer.attach([rp])

print("Writer attached. Output dir:", output_dir)

# ---------------------------------------------------------------------
# Main loop: step world + render frames
# ---------------------------------------------------------------------
frame_idx = 0

while simulation_app.is_running() and frame_idx < target_frames:
    if world.is_playing():
        if frame_idx % 10 == 0:
            print(f"Frame {frame_idx}...")
        world.step(render=True)
        frame_idx += 1

print(f"Completed {frame_idx} frames.")
simulation_app.close()
