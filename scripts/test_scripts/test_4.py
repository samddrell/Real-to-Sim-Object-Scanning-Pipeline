from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import os
import numpy as np

import omni.replicator.core as rep
from isaacsim.core.api import World
from isaacsim.core.prims import RigidPrim
from isaacsim.core.utils.prims import define_prim
from isaacsim.core.api.objects import DynamicSphere
from isaacsim.core.cloner import GridCloner


# ---------------------------------------------------------------------
# World + simple scene
# ---------------------------------------------------------------------
# create the world
world = World(stage_units_in_meters=1.0, physics_prim_path="/physicsScene", backend="numpy")
world.scene.add_default_ground_plane()

# set up grid cloner
cloner = GridCloner(spacing=1.5)
cloner.define_base_env("/World/envs")
define_prim("/World/envs/env_0")

# set up the first environment
DynamicSphere(prim_path="/World/envs/env_0/object", radius=0.1, position=np.array([0.75, 0.0, 0.2]))

# clone environments
num_envs = 4
prim_paths = cloner.generate_paths("/World/envs/env", num_envs)
env_pos = cloner.clone(source_prim_path="/World/envs/env_0", prim_paths=prim_paths)

# creates the views and set up world
object_view = RigidPrim(prim_paths_expr="/World/envs/*/object", name="object_view")
world.scene.add(object_view)
world.reset()
world.play()
print("WORLD PLAYING?", world.is_playing())

# ---------------------------------------------------------------------
# Add a camera + render product + PNG writer
# ---------------------------------------------------------------------

# Create a camera
camera = rep.create.camera(
    position=(2, 2, 2),
    look_at="/World/envs/env_0/object"
)

# Create a render product (used to render frames)
rp = rep.create.render_product(camera, resolution=(640, 480))

# Create a writer to output PNG files
writer_type = "BasicWriter"
writer = rep.WriterRegistry.get(writer_type)
# Make output directory absolute (optional but nice)
output_dir = os.path.join(os.getcwd(), "output_images_2")   # NOTE: I moved this folder to a data\datasets\output_images_2 for consistency
os.makedirs(output_dir, exist_ok=True)

writer.initialize(
    output_dir=output_dir,
    rgb=True
    # you can add other kwargs later like:
    # bounding_box_2d_tight=True,
    # semantic_segmentation=True,
)

# Attach the render product to the writer
writer.attach([rp])  # writer.attach(rp) also works in some versions
print("Writer attached. Output dir:", output_dir)

# ---------------------------------------------------------------------
# Main loop (stepping world + rendering)
# ---------------------------------------------------------------------

# We'll control how many frames by stopping our loop after 50 frames
target_frames = 50

frame_idx = 0
while simulation_app.is_running() and frame_idx < target_frames:
    if world.is_playing():

        # step physics and render
        world.step(render=True)

        frame_idx += 1

print(f"Completed {frame_idx} frames")

# Optional: close sim once done
simulation_app.close()
