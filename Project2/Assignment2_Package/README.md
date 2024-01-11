# CMSC848F Assignment 2: Single View to 3D
# Author : Rishikesh Avinash Jadhav(119256534)

## NOTE: The following folders contain the visualization results
f1_score plots - Contains the F1score Vs threshold graphs for voxels, point clouds, meshes for n_points = 5000 and n_points=10000
fitting_voxel - contains  GIFs demonstrating the model's voxel fitting process between source and target voxels
fitting_point - contains GIFs illustrating the model's point cloud fitting between source and target point clouds.
fitting_mesh - contains  GIFs demonstrating the model's mesh fitting process between source and target meshes.
rgb_images - contains the RGB images for comaprison with decoder outputs
image_voxel - contains the predicted and ground truth GIFs of the voxel decoder
image_point_cloud - contains the predicted and ground truth GIFs of the point cloud decoder
image_mesh - contains the predicted and ground truth GIFs of the mesh decoder
interpreting_model - contains the visualizations of activations of the second-to-last layer within the neural network's encoder. 

## 0. Setup

Please download and extract the dataset from [here](https://drive.google.com/file/d/1VoSmRA9KIwaH56iluUuBEBwCbbq3x7Xt/view?usp=sharing).
After unzipping, set the appropiate path references in `dataset_location.py` file [here](https://github.com/848f-3DVision/assignment2/blob/main/dataset_location.py#L2)

```
# Better do this after you've secured a GPU.
conda create -n pytorch3d-env python=3.9
conda activate pytorch3d-env
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d
pip install numpy PyMCubes matplotlib
```

Make sure you have installed the packages mentioned in `requirements.txt`.
This assignment will need the GPU version of pytorch.

[How to use GPUs on UMIACS cluster?](https://wiki.umiacs.umd.edu/umiacs/index.php/ClassAccounts#Cluster_Usage)

## 1. Exploring loss functions (15 points)

### 1.1. Fitting a voxel grid (5 points)

Run the file `python fit_data.py --type 'vox'`, to fit the source voxel grid to the target voxel grid. 

Run the file `python fit_data.py --type 'point'`, to fit the source point cloud to the target point cloud. 

Run the file `python fit_data.py --type 'mesh'`, to fit the source mesh to the target mesh. 

**Visualize the optimized mesh along-side the ground truth mesh using the tools learnt in previous section.**

## 2. Reconstructing 3D from single view (85 points) 
### NOTE: Every 50th Image is visualized in all cases

Run the file `python train_model.py --type 'vox'`

After trained, visualize the input RGB, ground truth voxel grid and predicted voxel in `eval_model.py` file using:
`python eval_model.py --type 'vox' --load_checkpoint`

### 2.2. Image to point cloud (20 points)
Run the file `python train_model.py --type 'point'`, to train single view to pointcloud pipeline, feel free to tune the hyperparameters as per your need.

After trained, visualize the input RGB, ground truth point cloud and predicted  point cloud in `eval_model.py` file using:
`python eval_model.py --type 'point' --load_checkpoint`


### 2.3. Image to mesh (20 points)

Run the file `python train_model.py --type 'mesh'`

After trained, visualize the input RGB, ground truth mesh and predicted mesh in `eval_model.py` file using:
`python eval_model.py --type 'mesh' --load_checkpoint`


### 2.4. Quantitative comparisions(10 points)
For evaluating you can run:
`python eval_model.py --type voxel|mesh|point --load_checkpoint`
Check web page for proper comparisons

### 2.5. Analyse effects of hyperparms variations (5 points)
I have changed the number of points from 5000, to 10000 and max_iterations= 2000, 5000. The repective plots for training are in the root repo.

### 2.6. Interpret your model (10 points)

Run the file `python interpreting_model.py` to check the visualizations for the second last layer feature maps of the encoder for the given 13 images (Every 50 images, one image is visualized)


