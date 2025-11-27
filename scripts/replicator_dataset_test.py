from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni.replicator.core as rep
from pathlib import Path
import os

# Resolve paths relative to this script's location
ROOT = Path(__file__).resolve().parents[1]   # go up from scripts/ to project root
PHOTO_DIR = ROOT / "data" / "photos_raw"
OUTPUT_DIR = ROOT / "data" / "datasets" / "photo_test"

IMAGE_PATHS = [
    str(p) for p in PHOTO_DIR.iterdir()
    if p.suffix.lower() in (".jpg", ".jpeg", ".png")
]

if not IMAGE_PATHS:
    raise RuntimeError(f"No images found in {PHOTO_DIR}")

with rep.new_layer():
    rep.create.plane(scale=50)
    camera = rep.create.camera(position=(0, 1, 2), look_at=(0, 0, 0))
    render_product = rep.create.render_product(camera, (640, 480))
    rep.create.light(light_type="Dome", intensity=8000)

    cards = []
    for img in IMAGE_PATHS:
        card = rep.create.plane(
            semantics=[("class", "photo")],
            position=(0, 0, 0),
            scale=1.0,
        )
        mat = rep.create.material_omniPBR()
        mat.set_texture("diffuse_texture", img)
        card.set_material(mat)
        cards.append(card)

    @rep.randomizer
    def photo_randomizer():
        card = rep.distribution.choice(cards)
        with card:
            rep.modify.pose(
                position=rep.distribution.uniform((-0.5, 0.3, -1.0), (0.5, 0.8, -2.0)),
                rotation=rep.distribution.uniform((0, -45, 0), (0, 45, 0)),
                scale=rep.distribution.uniform(0.5, 1.2),
            )

    photo_randomizer.register(trigger=rep.trigger.OnFrame())

    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir=str(OUTPUT_DIR),
        rgb=True,
        bounding_box_2d_tight=True,
    )
    writer.attach([render_product])

rep.orchestrator.run_until_complete(30)
simulation_app.close()
