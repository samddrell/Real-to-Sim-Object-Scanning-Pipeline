from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni.replicator.core as rep
from pathlib import Path

# ---------- Paths / setup ----------
ROOT = Path(__file__).resolve().parents[1]   # project root (up from scripts/)
OUTPUT_DIR = ROOT / "data" / "datasets" / "cube_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Replicator cube test starting.")
print(f"OUTPUT_DIR = {OUTPUT_DIR}")

# ---------- Build scene ----------
with rep.new_layer():

    # Ground plane
    rep.create.plane(scale=50, semantics=[("class", "ground")])

    # Simple cube object
    cube = rep.create.cube(
        semantics=[("class", "cube")],
        position=(0, 0.5, -2.0),
        scale=0.5,
    )

    # Camera looking at the cube
    camera = rep.create.camera(position=(0, 1, 1), look_at=(0, 0.5, -2.0))
    render_product = rep.create.render_product(camera, (640, 480))

    # Light
    rep.create.light(light_type="Dome", intensity=8000)

    # Single card that we’ll keep re-texturing
    card = rep.create.plane(
        semantics=[("class", "photo")],
        position=(0, 0, -1.5),
        scale=1.0,
    )

    # ---------- Randomization loop ----------
    # For 30 frames, randomize the card’s texture and pose
    with rep.trigger.on_frame(num_frames=30):
        with card:
            # Random texture from your PHOTO_DIR images
            rep.randomizer.texture(
                textures=IMAGE_PATHS,
                project_uvw=True,
            )
            # Random pose
            rep.modify.pose(
                position=rep.distribution.uniform(
                    (-0.5, 0.3, -2.0),   # z = -2.0 (lower)
                    (0.5, 0.8, -1.0),   # z = -1.0 (upper)
                ),
                rotation=rep.distribution.uniform(
                    (0, -45, 0),
                    (0, 45, 0),
                ),
                scale=rep.distribution.uniform(0.5, 1.2),
            )

    # ---------- Writer ----------
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir=str(OUTPUT_DIR),
        rgb=True,
        bounding_box_2d_tight=True,
    )
    writer.attach([render_product])

# Run until our on_frame trigger is done
rep.orchestrator.run()
simulation_app.close()
