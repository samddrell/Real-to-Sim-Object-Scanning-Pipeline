# Real-to-Sim-Object-Scanning-Pipeline

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
    - To run COLMAP in isolation, run the following commands:

```
cd C:\Users\samdd\Documents\school\ML\Real-to-Sim-Object-Scanning-Pipeline

C:\isaac-sim\python.bat scripts\third_party\colmap2nerf_compat.py `
  --run_colmap `
  --colmap_matcher exhaustive `
  --aabb_scale 16 `
  --images "C:\Users\samdd\Documents\school\ML\Real-to-Sim-Object-Scanning-Pipeline\data\bottle01\images" `
  --out "C:\Users\samdd\Documents\school\ML\Real-to-Sim-Object-Scanning-Pipeline\data\bottle01\transforms_auto.json" `
  --colmap_db "C:\Users\samdd\Documents\school\ML\Real-to-Sim-Object-Scanning-Pipeline\data\bottle01\colmap\auto_database.db" `
  --text "C:\Users\samdd\Documents\school\ML\Real-to-Sim-Object-Scanning-Pipeline\data\bottle01\colmap\auto_sparse" `
  --colmap_camera_model PINHOLE
```
### A Note on COLMAP
The file we use to produce camera data for our nerf in instant ngp is from the instant ngp library. It, unfortunately, is built around an old version of COLMAP.
Because the full COLMAP application must be installed to run this code, a few changes have been made to the file "colmap2nerf_compat.py" to avoid having to install
an old version of COLMAP. COLMAP 3.8+ now requires valid parameter lists for many camera models, so we have patched in two specific ways: 1) we changed the default camera 
to PINHOLE instead of OPENCV, and 2) we only include camera_params if the user provides them. The changed file has been provided for your convenience, and the relevant 
revisions have been indicated via comments on lines 36 and 125.