from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.prims import RigidPrim
from isaacsim.core.utils.prims import define_prim
from isaacsim.core.api.objects import DynamicSphere
from isaacsim.core.cloner import GridCloner

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


# set up randomization with isaacsim.replicator, imported as dr
import isaacsim.replicator.domain_randomization as dr
import omni.replicator.core as rep

dr.physics_view.register_simulation_context(world)
dr.physics_view.register_rigid_prim_view(object_view)

with dr.trigger.on_rl_frame(num_envs=num_envs):
    with dr.gate.on_interval(interval=20):
        dr.physics_view.randomize_simulation_context(
            operation="scaling",
            gravity=rep.distribution.uniform((1, 1, 0.0), (1, 1, 2.0)),
        )
    with dr.gate.on_interval(interval=50):
        dr.physics_view.randomize_rigid_prim_view(
            view_name=object_view.name,
            operation="direct",
            force=rep.distribution.uniform((0, 0, 2.5), (0, 0, 5.0)),
        )
    with dr.gate.on_env_reset():
        dr.physics_view.randomize_rigid_prim_view(
            view_name=object_view.name,
            operation="additive",
            position=rep.distribution.normal((0.0, 0.0, 0.0), (0.2, 0.2, 0.0)),
            velocity=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )


# ---------------------------------------------------------------------
# Add a camera + render product + PNG writer
# ---------------------------------------------------------------------
import omni.replicator.core as rep
import os

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

# We'll control how many frames by stopping our loop after 50 frames
target_frames = 50

frame_idx = 0
while simulation_app.is_running() and frame_idx < target_frames:
    if world.is_playing():
        # trigger resets every 200 steps
        reset_inds = []
        if frame_idx % 200 == 0:
            reset_inds = np.arange(num_envs)

        # advance DR graph for this RL step
        # This triggers the on_rl_frame which handles rendering and writing
        dr.physics_view.step_randomization(reset_inds)

        # step physics and render
        world.step(render=True)

        frame_idx += 1

print(f"Completed {frame_idx} frames")

# Optional: close sim once done
simulation_app.close()
