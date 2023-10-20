import torch
from pytorch3d.loss import mesh_laplacian_smoothing
from pytorch3d.ops.knn import knn_gather, knn_points

# define losses

# This loss function is used for fitting 3D binary voxel grids
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d

    # Apply the sigmoid activation function to squash values between 0 and 1
	sigmoid_activation = torch.nn.Sigmoid() #squash values between 0 and 1

    # Define the loss function as Binary Cross-Entropy (BCE) loss
	loss_function = torch.nn.BCELoss() 

    # Compute the loss by comparing the sigmoid-activated voxel_src to the target voxel_tgt\
	loss = loss_function(sigmoid_activation(voxel_src),voxel_tgt) # BCE between sigmoid activated voxel_src and voxel_tgt
	# implement some loss for binary voxel grids
	return loss

# This loss function is used for fitting 3D point clouds
def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	
    # Compute the number of points in the source and target point clouds
	source_points = torch.full((point_cloud_src.shape[0],), point_cloud_src.shape[1], dtype=torch.int64, device=point_cloud_src.device)
	target_ponts = torch.full((point_cloud_tgt.shape[0],), point_cloud_tgt.shape[1], dtype=torch.int64, device=point_cloud_tgt.device)

	# Use the knn_points function to find the nearest neighbors between points in the source and target point clouds
	source_to_target_nn = knn_points(point_cloud_src, point_cloud_tgt, lengths1=source_points, lengths2=target_ponts, norm=2, K=1)
	target_to_source_nn  = knn_points(point_cloud_tgt, point_cloud_src, lengths1=target_ponts, lengths2=source_points, norm=2, K=1)

	# Extract the distances to the nearest neighbors and sum them along the first dimension
	distances_source_to_target  = source_to_target_nn.dists[..., 0].sum(1)
	distances_target_to_source = target_to_source_nn.dists[..., 0].sum(1)

	# Calculate the Chamfer Loss by taking the mean of the sum of distances in both directions
	loss_chamfer = torch.mean(distances_source_to_target + distances_target_to_source)
	# implement chamfer loss from scratch
	return loss_chamfer

# This loss function is used for fitting meshes.
def smoothness_loss(mesh_src):
	loss_laplacian = mesh_laplacian_smoothing(mesh_src)
	# implement laplacian smoothening loss
	return loss_laplacian