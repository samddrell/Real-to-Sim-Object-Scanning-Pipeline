from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import os
import argparse
import numpy as np

from isaacsim.core.api import World
import omni.replicator.core as rep
import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdShade



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
# World + default ground plane
# ---------------------------------------------------------------------
world = World(
    stage_units_in_meters=1.0,
    physics_prim_path="/physicsScene",
    backend="numpy",
)
world.scene.add_default_ground_plane()   # ✅ back to the regular ground
world.reset()
world.play()
print("WORLD PLAYING?", world.is_playing())


# ---------------------------------------------------------------------
# Helper: create a SphereLight prim we can randomize
# ---------------------------------------------------------------------
def create_key_light(stage):
    prim_type = "SphereLight"
    light_path = omni.usd.get_stage_next_free_path(stage, "/World/KeyLight", False)
    light_prim = stage.DefinePrim(light_path, prim_type)

    xf = UsdGeom.Xformable(light_prim)
    xf.AddTranslateOp().Set((0.0, 0.0, 2.0))
    xf.AddScaleOp().Set((1.0, 1.0, 1.0))

    # Set up light attributes with some defaults
    light_prim.CreateAttribute("inputs:enableColorTemperature", Sdf.ValueTypeNames.Bool).Set(True)
    light_prim.CreateAttribute("inputs:colorTemperature", Sdf.ValueTypeNames.Float).Set(6500.0)
    light_prim.CreateAttribute("inputs:radius", Sdf.ValueTypeNames.Float).Set(0.5)
    light_prim.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(30000.0)
    light_prim.CreateAttribute("inputs:color", Sdf.ValueTypeNames.Color3f).Set((1.0, 1.0, 1.0))
    light_prim.CreateAttribute("inputs:exposure", Sdf.ValueTypeNames.Float).Set(0.0)
    light_prim.CreateAttribute("inputs:diffuse", Sdf.ValueTypeNames.Float).Set(1.0)
    light_prim.CreateAttribute("inputs:specular", Sdf.ValueTypeNames.Float).Set(1.0)

    return light_prim

stage = omni.usd.get_context().get_stage()
key_light_prim = create_key_light(stage)
print("Created key light at:", key_light_prim.GetPath())

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
        position=(0.0, 0.0, 2),
        # You can tweak this if your asset is off-center or rotated oddly
        rotation=(90.0, 0.0, 0.0),
    )

print("Loaded subject from USD.")

# ---------------------------------------------------------------------
# Camera + render product
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

# ---------------------------------------------------------------------
# Grab the underlying camera prim & base transform
# ---------------------------------------------------------------------
cam_prims_info = camera.get_output_prims()
cam_prim = cam_prims_info["prims"][0]
print("Camera prim path:", cam_prim.GetPath())

# Read base translate/rotate so we can jitter around them
base_cam_translate = cam_prim.GetAttribute("xformOp:translate").Get()
base_cam_rotate = cam_prim.GetAttribute("xformOp:rotateXYZ").Get()

if base_cam_translate is None:
    base_cam_translate = Gf.Vec3d(2.0, 2.0, 2.0)
if base_cam_rotate is None:
    base_cam_rotate = Gf.Vec3f(0.0, 0.0, 0.0)

base_cam_translate = np.array(list(base_cam_translate), dtype=float)
base_cam_rotate = np.array(list(base_cam_rotate), dtype=float)


# ---------------------------------------------------------------------
# PNG writer
# ---------------------------------------------------------------------

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

        # --- Randomize light attributes each frame ---

        # Position: jitter around the object
        tx = np.random.uniform(-2.0, 2.0)
        ty = np.random.uniform(-2.0, 2.0)
        tz = np.random.uniform(1.0, 4.0)
        key_light_prim.GetAttribute("xformOp:translate").Set((tx, ty, tz))

        # Scale (optional)
        scale_rand = np.random.uniform(0.5, 1.5)
        key_light_prim.GetAttribute("xformOp:scale").Set((scale_rand, scale_rand, scale_rand))

        # Color temperature (in Kelvin), with clamping
        temp = np.random.normal(4500.0, 1500.0)
        temp = float(np.clip(temp, 2000.0, 9000.0))
        key_light_prim.GetAttribute("inputs:colorTemperature").Set(temp)

        # Intensity, keep reasonably bright
        intensity = np.random.normal(25000.0, 5000.0)
        intensity = float(max(intensity, 1000.0))
        key_light_prim.GetAttribute("inputs:intensity").Set(intensity)

        # RGB color jitter
        color = (
            float(np.random.uniform(0.6, 1.0)),
            float(np.random.uniform(0.6, 1.0)),
            float(np.random.uniform(0.6, 1.0)),
        )
        key_light_prim.GetAttribute("inputs:color").Set(color)

        # --- Randomize OBJECT pose each frame (position + rotation) ---

        # Small translation jitter around base (0, 0, 1)
        pos_jitter_obj = np.array([
            np.random.uniform(-0.05, 0.05),   # X jitter
            np.random.uniform(-0.05, 0.05),   # Y jitter
            np.random.uniform(-0.02, 0.02),   # Z jitter
        ])
        base_obj_pos = np.array([0.0, 0.0, 0.3], dtype=float)
        new_obj_pos = base_obj_pos + pos_jitter_obj

        # Full 360° random rotation on each axis
        new_obj_rot = (
            float(np.random.uniform(0.0, 360.0)),  # X
            float(np.random.uniform(0.0, 360.0)),  # Y
            float(np.random.uniform(0.0, 360.0)),  # Z
        )

        with subject:
            rep.modify.pose(
                position=tuple(new_obj_pos.tolist()),
                rotation=new_obj_rot,
            )


        # --- randomize CAMERA pose each frame ---

        # Desired distance band from the object (in XY plane)
        r_min, r_max = 2.5, 4.5

        # Bias towards larger radii using sqrt-trick
        if frame_idx % 3 == 0:      # Reduce frequency of camera moves to increase speed
            # randomize object + camera
            u = np.random.uniform(0.0, 1.0)
            r = r_min + (r_max - r_min) * np.sqrt(u)

        theta = np.random.uniform(0.0, 2.0 * np.pi)

        cam_x = r * np.cos(theta)
        cam_y = r * np.sin(theta)
        cam_z = np.random.uniform(1.5, 3.0)  # height band

        # Re-apply look_at each frame so we never lose the subject
        with camera:
            rep.modify.pose(
                position=(float(cam_x), float(cam_y), float(cam_z)),
                look_at=subject,
            )


        # Small tilt perturbations (XY)
        rot_jitter = np.array([
            np.random.uniform(-5.0, 5.0),   # tilt in X (pitch-ish)
            np.random.uniform(-5.0, 5.0),   # tilt in Y (yaw-ish)
            np.random.uniform(-3.0, 3.0),   # small roll
        ])


        if frame_idx % 10 == 0:
            print(f"Frame {frame_idx}...")

        # Step sim+render
        world.step(render=True)
        frame_idx += 1

print(f"Completed {frame_idx} frames.")
simulation_app.close()
