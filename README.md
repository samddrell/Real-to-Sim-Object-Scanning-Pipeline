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