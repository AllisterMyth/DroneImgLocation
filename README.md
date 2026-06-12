**Introduction**

The goal of the project was to create a pipeline that takes a drone shot
and a satellite GeoTIFF to measure the location of the drone in
longitude/latitude. This is useful in GPS denied zones. Cross view
localisation is matching a photo taken from any perspective (In our case
upright) and matching it to existing satellite shots. There are many
mathematical algorithms like SIFT,SURF etc that have been the standard
for many years but Deep Learning models are more commonly used now.

**Methodology**

Model: I tried the classical models: USURF(upright SURF) and ORB first
and while they were okay time wise, they were not very accurate.

Next I used LoFTR and efficient LoFTR but they struggled with inlier
count possibly due to scale invariance in the images. Trying multiscale
image pyramid approach slowed them down much.

I went for feature based models for being more robust to scale variance. Tried Superpoint, Superglue but finally settled on Lightglue due to its much faster compute.

The pipeline has two main parts:

Precompute: The tif file is too heavy to calculate on the fly so I
decided to divide it into tiles with 25% overlap so that each feature
appears at least once.

The superpoint features and DINOv2 fingerprint vectors for the tif are
computed and saved before running the main script to save compute. The
descriptors are saved in float32.

Main: This currently works on MPS but I will add an option for CUDA.

This first loads the models and the descriptors, gps coordinates saved
by precompute. If the previous coordinates of the drone are unknown, It
then takes a drone image, grayscales it and uses superpoint and Dino on
them to get local descriptors and global fingerprint.

They are compared to the tif tile global fingerprints to find top 10
matches which are then compared by Lightglue. If no matches are found
the image is rotated 90 degrees and tried again.

But if previous coordinates are known then it just tries the surrounding
8 tiles first for matches to save compute. These are marked as local
matches and if no matches are found, then it goes to global searching.

For homography, I have used MAGSAC as its more accurate
than RANSAC. A 4 pixel deviation is allowed for high accuracy.

Geopositioning Calculation: The drone camera's centre pixel
is calculated and then after applying the homography matrix is
transformed into a pixel coordinates for a specific tile.

The top left pixel offset for that specific tile is taken from the
metadata and added to the pixel coordinates to make global coordinates.

Affine matrix is used to convert these coordinates to projected
coordinates which are then transformed into Longitude/ Latitude by using
CRS.

**Changes I made recently:**

I added CLAHE for better predictions in low inlier images.

I changed the quality to flat16 trading about 15% less inlier counts for
nearly half time reduction. The resolution is still kept at 1024px.

I added a second layer of scale 0.5x extracted for each image when local
fails and then when global fails.

I changed the order in which the rotations are checked now they take the
last known rotation, then check sideways and then backwards.

Memory optimisation is a big issue for both my laptop and the future
drone hardware so I added code to clean the descriptors and images after
every image.

This did not work so I tried directly adding more aggressive cleaning,
using tile cache, tensor buffering and more precompute.

This did not work due to PyTorch's garbage cleaners being
bad so I gave up on optimising the script and made a new stateless
script that runs this script as a subprocess in a loop for a certain
batch of images. This forces the os to clean up after every batch is
processed eliminating memory accumulation.

The main script saves the last result if existing as a vector which is
used by the next iteration. (Changed from idx to int for json)

Better solution is needed as the script needs to be loaded every time
its run in a loop which takes time (maybe run it async?)

Now the most memory I have seen it take is nearly 6-7gb but usually
stays around 3 gb.

I tried adding an altitude scaler as that would half the time it takes
for 0.5 scale results and provide more accuracy in general. Having
trouble getting the scale right, after taking alt from ground truth.

Alt scaling does not work when alt is measured from the previous image
as that can oscillate increasingly into huge deltas.

Tried adding IMU as well but that needs very precise timestamps
otherwise it hinders the rotation and priority of the tile processing
than benefit it. (Adding compass will be extremely helpful for rotation
priority)

Changed the csv writing code so that now it writes live when the script
is running to preserve data during crashes. It appends data it gets from
every loop.

Also added an option for cuda if mps is not available.

Removed warnings for mps fallback.

**Issues:**

This works bad or doesn't work for low altitudes. Can be
solved to some extent or completely by adding a real time altitude
scaling function.

Execution time is much for failed searches after a local search as it
first searches local on two scales and then checks global on two scales
(Maybe reduce the rotations to 2 instead of 4). This will also benefit
from the alt scaling function as then only one scale would be used so
cutting this time by half.

Running the Pipeline:

1. Create a python env and use pip install -r requirements.txt ; this will
help install all libraries and models needed for the pipeline.

2. Run pcom.py after saving map.tif in the same directory. This will
generate offline_assets file which contains the tiles, fingerprints
metadata, map parameters and features.

3. Put the images in a folder named Droneimg. Run run.py ; this will run the main.py as a subprocess over and over until all images are done. You may change the size of each batch in the script.

4. It will generate a localization_results.csv with the results in it. This csv will contain filename, latitude, longitude, altitude, inliers, mode and success or fail result. This can be edited in the csv writer in main.py.

**Evaluation & Scoring Rubric**

NOTE: This evaluation is done above 40m images as the model doesn't work properly on lower altitudes due to scaling issues. 

  ----------------------- ----------------------- -----------------------
  **Metric**              **Value**               **Notes /Observations**

                                                                                                       

  **Total Images Processed**         349          Images from 1421.3400
                                                  to 1593.7817.

  **Recall @ 5m (%)**     91.12%                  318 of 349 images had
                                                  an error \< 5 meters.

  **Recall @ 20m (%)**    95.70%                  334 of 349 images had
                                                  an error \< 20 meters.

  **Precision @ 5m (%)**  91.38%                  Out of 348 registered
                                                  images, 318 were
                                                  accurate within 5m.

  **Precision @ 20m (%)** 95.98%                  Out of 348 registered
                                                  images, 334 were
                                                  accurate within 20m.

  **Median Localisation Error**    2.84 m         Typical performance;
                                                  median is resilient to
                                                  outliers.

  **Mean Localisation Error**     8.12 m          Higher than median due
                                                  to \~4% outlier rate
                                                  (\>50m error).

  **Mode Localisation Error**      1.15 m         The most frequent error
                                                  cluster (highly precise
                                                  matches).

  **Error Variance**      24.81 m²                Moderate variance
                                                  caused by a few
                                                  \"catastrophic\" jumps.

  **Avg. Inlier Count**   41.2                    Average number of
                                                  points surviving MAGSAC
                                                  per image.

  **Registration Success Rate**  99.71%           348/349 images
                                                  successfully returned
                                                  coordinates.

  **Avg. Execution Time** 1.95 s                  Includes fast LOCAL
                                                  updates (\~1.4s) and
                                                  slower GLOBAL (\~5.4s).

  **Feature Extractor/ Descriptor Used**   Superpoint/Lightglue    
                                                                  

  **Matching Strategy**   Nearest neighbour       
  ----------------------- ----------------------- -----------------------
