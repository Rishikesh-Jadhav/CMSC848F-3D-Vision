# CMSC848F Assignment 2: Single View to 3D

Goals: In this assignment, you will explore the types of loss and decoder functions for regressing to voxels, point clouds, and mesh representation from single view RGB input. 

## 0. Setup

Please download and extract the dataset from [here](https://drive.google.com/file/d/1VoSmRA9KIwaH56iluUuBEBwCbbq3x7Xt/view?usp=sharing).
After unzipping, set the appropiate path references in `dataset_location.py` file [here](https://github.com/848f-3DVision/assignment2/blob/main/dataset_location.py#L2)

Make sure you have installed the packages mentioned in `requirements.txt`.
This assignment will need the GPU version of pytorch.

[How to use GPUs on UMIACS cluster?](https://wiki.umiacs.umd.edu/umiacs/index.php/ClassAccounts#Cluster_Usage)

## 1. Exploring loss functions (15 points)
This section will involve defining a loss function, for fitting voxels, point clouds and meshes.

### 1.1. Fitting a voxel grid (5 points)
In this subsection, we will define binary cross entropy loss that can help us <b>fit a 3D binary voxel grid</b>.
Define the loss functions [here](https://github.com/848f-3DVision/assignment2/blob/main/losses.py#L4-L9) in `losses.py` file. 
For this you can use the pre-defined losses in pytorch library.

Run the file `python fit_data.py --type 'vox'`, to fit the source voxel grid to the target voxel grid. 

**Visualize the optimized voxel grid along-side the ground truth voxel grid using the tools learnt in previous section.**

### 1.2. Fitting a point cloud (5 points)
In this subsection, we will define chamfer loss that can help us <b> fit a 3D point cloud </b>.
Define the loss functions [here](https://github.com/848f-3DVision/assignment2/blob/main/losses.py#L11-L15) in `losses.py` file.
<b>We expect you to write your own code for this and not use any pytorch3d utilities. You are allowed to use functions inside pytorch3d.ops.knn such as knn_gather or knn_points</b>

Run the file `python fit_data.py --type 'point'`, to fit the source point cloud to the target point cloud. 

**Visualize the optimized point cloud along-side the ground truth point cloud using the tools learnt in previous section.**

### 1.3. Fitting a mesh (5 points)
In this subsection, we will define an additional smoothening loss that can help us <b> fit a mesh</b>.
Define the loss functions [here](https://github.com/848f-3DVision/assignment2/blob/main/losses.py#L17-L20) in `losses.py` file.

For this you can use the pre-defined losses in pytorch library.

Run the file `python fit_data.py --type 'mesh'`, to fit the source mesh to the target mesh. 

**Visualize the optimized mesh along-side the ground truth mesh using the tools learnt in previous section.**

## 2. Reconstructing 3D from single view (85 points)
This section will involve training a single view to 3D pipeline for voxels, point clouds and meshes.
Refer to the `save_freq` argument in `train_model.py` to save the model checkpoint quicker/slower. 

We also provide pretrained ResNet18 features of images to save computation and GPU resources required. Use `--load_feat` argument to use these features during training and evaluation. This should be False by default, and only use this if you are facing issues in getting GPU resources. You can also enable training on a CPU by the `device` argument. Also indiciate in your submission if you had to use this argument. 

### 2.1. Image to voxel grid (20 points)
In this subsection, we will define a neural network to decode binary voxel grids.
Define the decoder network [here](https://github.com/848f-3DVision/assignment2/blob/main/model.py#L149) in `model.py` file, then reference your decoder [here](https://github.com/848f-3DVision/assignment2/blob/main/model.py#L182) in `model.py` file.

We have provided a [decoder network](https://github.com/848f-3DVision/assignment2/blob/main/model.py#L55-L129) in `model.py`, but you can also modify it as you wish. 

Run the file `python train_model.py --type 'vox'`, to train single view to voxel grid pipeline, feel free to tune the hyperparameters as per your need.

After trained, visualize the input RGB, ground truth voxel grid and predicted voxel in `eval_model.py` file using:
`python eval_model.py --type 'vox' --load_checkpoint`

You need to add the respective visualization code in `eval_model.py`

On your webpage, you should include visuals of any three examples in the test set. For each example show the input RGB, render of the predicted 3D voxel grid and a render of the ground truth mesh.

### 2.2. Image to point cloud (20 points)
In this subsection, we will define a neural network to decode point clouds.
Similar as above, define the decoder network [here](https://github.com/848f-3DVision/assignment2/blob/main/model.py#L155) in `model.py` file, then reference your decoder [here](https://github.com/848f-3DVision/assignment2/blob/main/model.py#L215) in `model.py` file

Run the file `python train_model.py --type 'point'`, to train single view to pointcloud pipeline, feel free to tune the hyperparameters as per your need.

After trained, visualize the input RGB, ground truth point cloud and predicted  point cloud in `eval_model.py` file using:
`python eval_model.py --type 'point' --load_checkpoint`

You need to add the respective visualization code in `eval_model.py`.

On your webpage, you should include visuals of any three examples in the test set. For each example show the input RGB, render of the predicted 3D point cloud and a render of the ground truth mesh.


### 2.3. Image to mesh (20 points)
In this subsection, we will define a neural network to decode mesh.
Similar as above, define the decoder network [here](https://github.com/848f-3DVision/assignment2/blob/main/model.py#L177) in `model.py` file, then reference your decoder [here](https://github.com/848f-3DVision/assignment2/blob/main/model.py#L220) in `model.py` file

Run the file `python train_model.py --type 'mesh'`, to train single view to mesh pipeline, feel free to tune the hyperparameters as per your need. We also encourage the student to try different mesh initializations [here](https://github.com/848f-3DVision/assignment2/blob/main/model.py#L174)


After trained, visualize the input RGB, ground truth mesh and predicted mesh in `eval_model.py` file using:
`python eval_model.py --type 'mesh' --load_checkpoint`

You need to add the respective visualization code in `eval_model.py`.

On your webpage, you should include visuals of any three examples in the test set. For each example show the input RGB, render of the predicted mesh and a render of the ground truth mesh.

### 2.4. Quantitative comparisions(10 points)
Quantitatively compare the F1 score of 3D reconstruction for meshes vs pointcloud vs voxelgrids.
**Provide an intutive explaination justifying the comparision.**

For evaluating you can run:
`python eval_model.py --type voxel|mesh|point --load_checkpoint`


On your webpage, you should include the f1-score curve at different thresholds for voxelgrid, pointcloud and the mesh network. The plot is saved as `eval_{type}.png`.

### 2.5. Analyse effects of hyperparms variations (5 points)
Analyse the results, by varying an hyperparameter of your choice.
For example `n_points` or `vox_size` or `w_chamfer` or `initial mesh(ico_sphere)` etc.
Try to be unique and conclusive in your analysis.

### 2.6. Interpret your model (10 points)
Simply seeing final predictions and numerical evaluations is not always insightful. Can you create some visualizations that help highlight what your learned model does? Be creative and think of what visualizations would help you gain insights. There is no `right' answer - although reading some papers to get inspiration might give you ideas.


