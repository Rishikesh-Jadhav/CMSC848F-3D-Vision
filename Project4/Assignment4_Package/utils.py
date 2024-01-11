import os
import torch
import pytorch3d
from pytorch3d.renderer import (
    AlphaCompositor,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
)
from PIL import Image, ImageDraw
import imageio
import numpy as np
def save_checkpoint(epoch, model, args, best=False):
    if best:
        path = os.path.join(args.checkpoint_dir, 'best_model.pt')
    else:
        path = os.path.join(args.checkpoint_dir, 'model_epoch_{}.pt'.format(epoch))
    torch.save(model.state_dict(), path)

def create_dir(directory):
    """
    Creates a directory if it does not already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_points_renderer(
    image_size=256, device=None, radius=0.01, background_color=(1, 1, 1)
):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.
    
    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer

def viz_cls (verts, path, device, gt_class,pred_class):
    """
    visualize classification result
    output: a 360-degree gif
    """
    image_size=256
    background_color=(1, 1, 1)

    # Construct various camera viewpoints
    dist = 3
    elev = 0
    azim = [180 - 12*i for i in range(30)]
    R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
    c = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)

    sample_verts = verts.repeat(30,1,1).to(torch.float)
    sample_colors = torch.tensor([0.7,0.3,1.0]).repeat(1,sample_verts.shape[1],1).repeat(30,1,1).to(torch.float)

    point_cloud = pytorch3d.structures.Pointclouds(points=sample_verts, features=sample_colors).to(device)

    renderer = get_points_renderer(image_size=image_size, background_color=background_color, device=device)
    rend = renderer(point_cloud, cameras=c).cpu().numpy() # (30, 256, 256, 3)
    class_map = {
        0: 'chair',
        1: 'vase',
        2: 'lamp'
    }
    # Convert and scale the image data if necessary
    if rend.dtype == np.float32:  # Check if the data type is float32
        rend = np.clip(rend, 0, 1)  # Ensure all values are within [0, 1]
        rend = (rend * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8
    gt_class= gt_class.item() if torch.is_tensor(gt_class) else gt_class
    pred_class = pred_class.item() if torch.is_tensor(pred_class) else pred_class

    images = []
    for i, r in enumerate(rend):
        image = Image.fromarray(r ,'RGB')
        draw = ImageDraw.Draw(image)
        draw.text((20, 20), f"Ground truth: {class_map[gt_class]}, Predicted: {class_map[pred_class]}", fill=(0, 0, 255))
        images.append(np.array(image))
    imageio.mimsave(path, images, fps=15)


def viz_seg (verts, labels, path, device,num_points):
    """
    visualize segmentation result
    output: a 360-degree gif
    """
    image_size=256
    background_color=(1, 1, 1)
    colors = [[1.0,1.0,1.0], [1.0,0.5,0.4], [0.0,1.0,1.0],[1.0,1.0,0.0],[0.0,0.0,1.0], [1.0,0.0,0.0]]

    # Construct various camera viewpoints
    dist = 3
    elev = 0
    azim = [180 - 12*i for i in range(30)]
    R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
    c = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)

    sample_verts = verts.unsqueeze(0).repeat(30,1,1).to(torch.float)
    sample_labels = labels.unsqueeze(0)
    sample_colors = torch.zeros((1,num_points,3))
    # Colorize points based on segmentation labels
    for i in range(6):
        sample_colors[sample_labels==i] = torch.tensor(colors[i])

    sample_colors = sample_colors.repeat(30,1,1).to(torch.float)

    point_cloud = pytorch3d.structures.Pointclouds(points=sample_verts, features=sample_colors).to(device)

    renderer = get_points_renderer(image_size=image_size, background_color=background_color, device=device)
    rend = renderer(point_cloud, cameras=c).cpu().numpy() # (30, 256, 256, 3)

    # Convert and scale the image data if necessary
    if rend.dtype == np.float32:  # Check if the data type is float32
        rend = np.clip(rend, 0, 1)  # Ensure all values are within [0, 1]
        rend = (rend * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8
    imageio.mimsave(path, rend, fps=15)
    
def rotate_point_cloud(test_dataloader):
    """
    Rotates the point cloud by fixed angles around each axis.
    :return: rotated point cloud as a numpy array
    """
    rot = torch.tensor([90,0,0], dtype=torch.float)
    rot = torch.deg2rad(rot)
    R = pytorch3d.transforms.euler_angles_to_matrix(rot, 'XYZ')
    test_dataloader.dataset.data = (R @ test_dataloader.dataset.data.transpose(1, 2)).transpose(1, 2)
    rad = torch.Tensor([10 * np.pi / 180.])[0]

    R_x = torch.Tensor([[1, 0, 0],
                        [0, torch.cos(rad), - torch.sin(rad)],
                        [0, torch.sin(rad), torch.cos(rad)]])
    R_y = torch.Tensor([[torch.cos(rad), 0, torch.sin(rad)],
                        [0, 1, 0],
                        [- torch.sin(rad), 0, torch.cos(rad)]])
    R_z = torch.Tensor([[torch.cos(rad), - torch.sin(rad), 0],
                        [torch.sin(rad), torch.cos(rad), 0],
                        [0, 0, 1]])

    test_dataloader.dataset.data = ((R_x @ R_y @ R_z) @ test_dataloader.dataset.data.transpose(1, 2)).transpose(1, 2)
    
    return test_dataloader