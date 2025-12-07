# Real-to-Sim-Object-Scanning-Pipeline

## Directory Structure
bottle01 is provided as an example.
bottle02 is provided as an opportunity to replicate the example.

## Steps:
### 1. Take 50+ photos of object
    - Need to be similarly framed with object at center of photo
    - simple background
    - On non-reflective surface that's clear of clutter
    - Circle the object from a variety of heights and angles. Each image should have large overlap.
    - Make sure the images are jpg's or png's (jpgs process faster due to compression)
    - Dont crop - this will happen at a later stage
### 2. Run COLMAP
    - This step extracts camera data that is crucial to nerf training
    - Produces a "transform.json" file that contains camera information for each image.
    - Will leave out photos it deems bad. A Good original photo sample set is crucial to ensure that each image is included.
    - To run COLMAP, run the following commands. Of course, customize file paths.

```
C:\isaac-sim\python.bat src\prepare_colmap_scene.py `
    "C:\Users\samdd\Documents\school\ML\Real-to-Sim-Object-Scanning-Pipeline\data\bottle02\images" `
    --scene_name bottle02
```

For your convenience, the code to run COLMAP directly from the commandline is provided below. Of course, customize file paths.

```
cd C:\Users\samdd\Documents\school\ML\Real-to-Sim-Object-Scanning-Pipeline

C:\isaac-sim\python.bat scripts\third_party\colmap2nerf_compat.py `
  --run_colmap `
  --colmap_matcher exhaustive `
  --aabb_scale 16 `
  --images "C:\Users\samdd\Documents\school\ML\Real-to-Sim-Object-Scanning-Pipeline\data\bottle02\images" `
  --out "C:\Users\samdd\Documents\school\ML\Real-to-Sim-Object-Scanning-Pipeline\data\bottle02\transforms_auto.json" `
  --colmap_db "C:\Users\samdd\Documents\school\ML\Real-to-Sim-Object-Scanning-Pipeline\data\bottle02\colmap\auto_database.db" `
  --text "C:\Users\samdd\Documents\school\ML\Real-to-Sim-Object-Scanning-Pipeline\data\bottle02\colmap\auto_sparse" `
  --colmap_camera_model PINHOLE
```
### A Note on COLMAP
The file we use to produce camera data for our nerf in instant ngp is from the instant ngp library. It, unfortunately, is built around an old version of COLMAP.
Because the full COLMAP application must be installed to run this code, a few changes have been made to the file "colmap2nerf_compat.py" to avoid having to install
an old version of COLMAP. COLMAP 3.8+ now requires valid parameter lists for many camera models, so we have patched in two specific ways: 1) we changed the default camera 
to PINHOLE instead of OPENCV, and 2) we only include camera_params if the user provides them. The changed file has been provided for your convenience, and the relevant 
revisions have been indicated via comments on lines 36 and 125.

### 3. Run Instant-NGP
This is the process two train a nerf from the provided images and from the transforms.json file that COLMAP produced. Run the GUI version of this program.
    - Once the loss graph has converged (a little oscillation is ok - the graph scales), end training.
    - Use the crop feature to isolate the target object (in the example case, the water bottle)
    - Build and export the mesh version of the object. This will save a .obj file of the object.
```
cd C:\Apps\Instant-NGP-for-RTX-2000
.\instant-ngp.exe --scene "C:\Users\samdd\Documents\school\ML\Real-to-Sim-Object-Scanning-Pipeline\data\bottle02"

```

### 4. Convert OBJ to USD
The ISAAC Replicator cannot build scenes from .obj files, so we must convert our mesh object to a .usd file. Fortunately, ISAAC does have this conversion feature built in.
To run the ISAAC converter, run the following code. Of course, customize file paths.

```
C:\isaac-sim\python.bat src\convert_obj_to_usd.py `
  --mesh "C:\Users\samdd\Documents\school\ML\Real-to-Sim-Object-Scanning-Pipeline\data\bottle02\transforms_base.obj" `
  --usd  "C:\Users\samdd\Documents\school\ML\Real-to-Sim-Object-Scanning-Pipeline\data\bottle02\bottle01.usd"
```

### A Note on Mesh Resolution
You may notice that the bottle's mesh is quite low resolution and chunky; the stickers on the sides are not nearly as clear in the mesh as they were in the nerf, and the bottle itself is not smooth. This may cause the yolo model to learn the irregular geometry of this bottle, but it also may cause it to learn less specific details that indicate that it is a bottle. This begs the question: what qualities must be present in a synthetic dataset to improve object detection? That is the question we seek to address.

### 5. Run Isaac Replicator script
```
# From Repo Root <path>\Real-to-Sim-Object-Scanning-Pipeline>
# NOTE: The frame count can be minized for faster test runs.
C:\isaac-sim\python.bat `
   .\scripts\test_scripts\replicator_subject_test.py `
   --usd .\data\bottle02\bottle02.usd `
   --frames 3000 `
   --out "E:\synthetic_datasets\bottle02"
```

### 6. Run baseline YOLO test
- Label dataset, bottle01 is provided for reference. bottle01_real_only.yaml is also provided as an example. Use https://www.makesense.ai/ to easily label dataset.
- Need conda
- need python enviornment
```
# In Anaconda PowerShell
conda activate yolo

# Then cd into your repo
cd C:\Users\samdd\Documents\school\ML\Real-to-Sim-Object-Scanning-Pipeline

# Run the script with that env's python
python .\src\baseline_yolo_control.py
```
- COMMENT ON TEST FILE LOCATION OR FIND A WAY TO INCLUDE THESE IN REPO WITHOUT HAVING THEM ON MY MACHINE

### Baseline Test Results
Good precision - when it detects bottle, there is bottle. Recall is low, this shows that it can't always find the bottle in the first place. Plainly, when it does find a bottle, it's right. Of course, every image in our test set includes the bottle. At first, it may seem like it would skew the recall higher (and maybe, in fact, it does), but because the recall is still so low, we still need to improve that score by training the model on the bottle in different conditions. This is what our synthetic dataset seeks to solve! 
        "trained_test": 
            "tag": "trained_test",
            "map50_95": 0.535544662232452,
            "map50": 0.7758627521447564,
            "map75": 0.6100017825360953,
            "precision": 0.9480671582561665,
            "recall": 0.61537068458545,
- INCLUDE CONFUSION MATRIX AND OTHER GRAPHS

## Section X: Domain Randomization Strategy for Synthetic Object Rendering

(Outline)

1. Introduction to Domain Randomization

Brief explanation of domain randomization (DR) as a technique to improve generalization from synthetic to real data.

Motivation for using DR in a NeRF/photogrammetry-based object-scanning pipeline.

Key insight: Since the object is static and no physical interaction occurs, DR focuses entirely on visual rather than physical variability.

2. Scope of Randomization in This Work

Summary statement describing what is randomized in the dataset:

lighting

camera pose

background

object pose

material microproperties

Clarification of what is intentionally not randomized:

physics simulation

object dynamics

distractor objects

articulated motion

Rationale: dataset goal is classification/segmentation/general appearance modeling, not control or manipulation.

3. Camera Pose Randomization
3.1 Motivation

Helps the model learn view-invariant representations of the object.

Ensures coverage across all possible viewpoints encountered during inference.

3.2 Parameters Randomized

Position distribution (distance, azimuth, elevation).

Look-at jitter.

Intrinsic variability (optional): focal length jitter, FOV.

3.3 Implementation Details

Description of random sampling distributions used in Replicator.

Camera frames per object per environment.

4. Lighting Randomization
4.1 Motivation

Lighting is one of the primary sources of domain shift between synthetic and real captures.

Prevents overfitting to NeRF’s learned environment lighting or photogrammetry’s baked reflections.

4.2 Parameters Randomized

Light positions.

Light intensity range.

Color temperature variation / RGB variation.

Number of active lights (optional).

Directionality (spotlight vs. area light selection if used).

4.3 Implementation Details

Use of Replicator light primitives.

Sampling distributions for intensity, color, and placement.

5. Background Randomization
5.1 Motivation

Background bias is a major contributor to synthetic-to-real performance drop.

Essential when training detection or segmentation models.

5.2 Types of Backgrounds Used

HDRI environment maps.

Solid-color backdrops.

Procedural noise or texture backgrounds.

Blurred natural images (optional).

5.3 Implementation Details

Random selection of backgrounds per frame.

Intensity controls for environment lighting, if applicable.

6. Object Pose Randomization
6.1 Motivation

Ensures robustness to real-world placement variability.

Helps in training object orientation estimation models (if relevant).

6.2 Parameters Randomized

Rotation around all axes.

Translation jitter within a bounded range.

Optional: small scale jitter (if allowed by research scope).

6.3 Implementation Details

Pose perturbations applied directly to the USD prim.

Uniform and Gaussian distributions depending on parameter type.

7. Material Property Randomization (Optional)
7.1 Motivation

Photogrammetry and NeRF reconstructions may inaccurately predict roughness/metalness maps.

Real objects exhibit variability due to wear, lighting, and sensor noise.

7.2 Parameters Randomized

Roughness.

Specular reflectance.

Metallic factor.

Base color tint.

7.3 Implementation Details

Random perturbed shader parameters applied per frame.

8. Summary of Randomization Strategy

Concise recap of the visual DR pipeline.

Justification of why these particular randomizations are aligned with the dataset’s intended downstream tasks.

Discussion of how this strategy avoids unnecessary or irrelevant DR components (e.g., physics, distractors).

Bridge sentence leading into the next section (e.g., dataset generation procedure, rendering pipeline, data export, or evaluation protocol).

Optional Figures/Illustrations

You may want to include:

Diagram of randomization components around the object.

Sample grid of images showing lighting/camera/background variation.

Visual comparison of with/without DR (small figure).