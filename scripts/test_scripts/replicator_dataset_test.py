from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni.replicator.core as rep
from pathlib import Path
import os

# ---------- Paths / setup ----------
ROOT = Path(__file__).resolve().parents[1]   # project root (up from scripts/)
OUTPUT_DIR = ROOT / "data" / "datasets" / "cube_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n=== Replicator cube test starting (instrumented) ===")
print(f"cwd             = {os.getcwd()}")
print(f"OUTPUT_DIR      = {OUTPUT_DIR}")
print(f"OUTPUT_DIR abs? = {OUTPUT_DIR.is_absolute()}")
print(f"OUTPUT_DIR exists? {OUTPUT_DIR.exists()}")

# For sanity, show all registered writers
print("Available writers in WriterRegistry:", rep.WriterRegistry.get_writers())

# ---------- Build scene ----------
with rep.new_layer():

    # ---------- Ground plane ----------
    print("Creating ground plane...")
    ground = rep.create.plane(
        scale=10,
        position=(0, 0, 0),
        semantics=[("class", "ground")],
    )
    # Light gray ground
    ground_mat = rep.create.material_omnipbr(
        diffuse=[0.6, 0.6, 0.6],
        roughness=0.4,
    )
    ground.set_material(ground_mat)

    # ---------- Cube ----------
    print("Creating cube...")
    cube = rep.create.cube(
        semantics=[("class", "cube")],
        position=(0, 0.5, -2.0),
        scale=0.5,
    )
    # Bright colored cube
    cube_mat = rep.create.material_omnipbr(
        diffuse=[0.9, 0.2, 0.2],   # red-ish
        roughness=0.3,
    )
    cube.set_material(cube_mat)


    print("Creating camera + render product...")
    camera = rep.create.camera(position=(0, 1, 1), look_at=(0, 0.5, -2.0))
    render_product = rep.create.render_product(camera, (640, 480))
    print(f"Render product: {render_product}")

    # ---------- Strong key light ----------
    print("Creating strong key light...")
    key_light = rep.create.light(
        light_type="Sphere",
        position=(0, 3.0, -2.0),   # above the cube
        intensity=500_000,         # MUCH brighter than before
        temperature=6500,          # daylight
        visible=False,             # don't render the light source itself
    )

    # ---------- Randomization loop ----------
    print("Setting up on_frame trigger (max_execs=5)...")
    with rep.trigger.on_frame(max_execs=5):
        with cube:
            rep.modify.pose(
                position=rep.distribution.uniform(
                    (-0.5, 0.3, -3.0),
                    (0.5, 0.8, -1.5),
                ),
                rotation=rep.distribution.uniform(
                    (0, -45, 0),
                    (0, 45, 0),
                ),
                scale=rep.distribution.uniform(0.4, 0.8),
            )

    # ---------- Writer ----------
    print("Getting BasicWriter from WriterRegistry...")
    writer = rep.WriterRegistry.get("BasicWriter")
    print(f"Writer object: {writer}")

    cfg = {
        "output_dir": str(OUTPUT_DIR),
        "rgb": True,
        "bounding_box_2d_tight": True,
    }
    print("Initializing writer with config:", cfg)
    writer.initialize(**cfg)

    print("Attaching writer to render_product...")
    writer.attach([render_product])

print("Triggers configured. Running orchestrator with run_until_complete()...")
rep.orchestrator.run_until_complete()
print("Orchestrator run complete.")

# ---------- Inspect OUTPUT_DIR from inside the app ----------
print("\nContents of OUTPUT_DIR after run:")
if OUTPUT_DIR.exists():
    any_files = False
    for p in OUTPUT_DIR.rglob("*"):
        print(" -", p, "(dir)" if p.is_dir() else "(file)")
        any_files = True
    if not any_files:
        print(" (no files or subdirectories found)")
else:
    print("OUTPUT_DIR does not exist (this would be weird).")

print("\nClosing SimulationApp...")
simulation_app.close()
print("SimulationApp closed.")
