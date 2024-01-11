import numpy as np
import argparse

import torch
from models import seg_model
from data_loader import get_data_loader
from utils import create_dir, viz_seg, rotate_point_cloud
from data_loader import get_data_loader

import random
import pytorch3d

# Set a random seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_seg_class', type=int, default=6, help='The number of segmentation classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model') 
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/seg/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/seg/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output/seg')

    parser.add_argument('--exp_num', type=int, default=0, help='The name of the experiment')

    parser.add_argument('--main_dir', type=str, default='./data/')
    parser.add_argument('--task', type=str, default="seg", help='The task: cls or seg')
    parser.add_argument('--batch_size', type=int, default=16, help='The number of images in a batch.')
    parser.add_argument('--num_workers', type=int, default=0, help='The number of threads to use for the DataLoader.')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Segmentation Task  ------
    model = seg_model().to(args.device)
    
    # Load Model Checkpoint
    model_path = './checkpoints/seg/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    test_label = torch.from_numpy((np.load(args.test_label))[:,ind])

    # ------ TO DO: Make Prediction ------

    # Load the test data using a data loader based on the provided arguments
    test_dataloader = get_data_loader(args=args, train=False)

    # If the experiment number is 1, perform point cloud rotation on the test data
    if args.exp_num == 1:
        test_dataloader = rotate_point_cloud(test_dataloader)

    # Initialize variables to track correct predictions and total number of points
    correct_point = 0
    num_point = 0

    # List to store predicted labels for visualization
    preds_labels = []

    # Iterate through batches in the test data loader
    for batch in test_dataloader:
        # Extract point clouds and labels from the batch
        point_clouds, labels = batch
        
        # Select a subset of points based on the 'ind' index and move to the specified device
        point_clouds = point_clouds[:, ind].to(args.device)
        
        # Select corresponding labels based on the 'ind' index and move to the specified device
        labels = labels[:, ind].to(args.device).to(torch.long)

        # Disable gradient computation during inference
        with torch.no_grad():
            # Get model predictions and find the index with maximum value along the last dimension
            pred_labels = torch.argmax(model(point_clouds), dim=-1, keepdim=False)
        
        # Update the count of correctly predicted points
        correct_point += pred_labels.eq(labels.data).cpu().sum().item()
        
        # Update the total number of points
        num_point += labels.view([-1, 1]).size()[0]

        # Store the predicted labels for later visualization
        preds_labels.append(pred_labels)

    # Calculate test accuracy based on the collected counts
    test_accuracy = correct_point / num_point

    # Print the test accuracy
    print(f"test accuracy: {test_accuracy}")

    # Concatenate the predicted labels and move to CPU for visualization
    preds_labels = torch.cat(preds_labels).detach().cpu()

    # Visualize a random subset (25 examples) of the predictions
    for i in range(25):
        # Randomly select an index from the predicted labels
        random_ind = random.randint(0, preds_labels.shape[0] - 1)

        # Extract point cloud and ground truth labels for visualization
        verts = test_dataloader.dataset.data[random_ind, ind].detach().cpu()
        labels = test_dataloader.dataset.label[random_ind, ind].to(torch.long).detach().cpu()

        # Calculate accuracy for the selected example
        correct_point = preds_labels[random_ind].eq(labels.data).cpu().sum().item()
        num_point = labels.view([-1, 1]).size()[0]
        accuracy = correct_point / num_point

        # Visualize the ground truth and predicted labels and save as GIF files
        viz_seg(verts, labels, "{}/random_vis_{}_gt_{}_acc{}.gif".format(args.output_dir, random_ind, args.exp_num, accuracy), args.device, args.num_points)
        viz_seg(verts, preds_labels[random_ind], "{}/pred_{}.gif".format(args.output_dir, random_ind), args.device,args.num_points)
