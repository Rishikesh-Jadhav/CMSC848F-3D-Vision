
import numpy as np
import argparse

import torch
from models import cls_model
from utils import create_dir, viz_cls, rotate_point_cloud
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

    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=1000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model') #model_epoch_0
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/cls/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output/cls')

    parser.add_argument('--exp_num', type=int, default=0, help='The number of the experiment')

    parser.add_argument('--main_dir', type=str, default='./data/')
    parser.add_argument('--task', type=str, default="cls", help='The task: cls or seg')
    parser.add_argument('--batch_size', type=int, default=16, help='The number of images in a batch.')
    parser.add_argument('--num_workers', type=int, default=0, help='The number of threads to use for the DataLoader.')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Classification Task ------
    model = cls_model().to(args.device)
    
    # Load Model Checkpoint
    model_path = './checkpoints/cls/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    test_label = torch.from_numpy(np.load(args.test_label))

    # ------ TO DO: Make Prediction ------

    # Load the test data using a data loader based on the provided arguments
    test_dataloader = get_data_loader(args=args, train=False)

    # If the experiment number is 1, perform point cloud rotation on the test data
    if args.exp_num == 1:
        test_dataloader = rotate_point_cloud(test_dataloader)

    # Initialize variables to track correct predictions and total number of objects
    correct_obj = 0
    num_obj = 0

    # List to store predicted labels for visualization
    preds_labels = []

    # Iterate through batches in the test data loader
    for batch in test_dataloader:
        # Extract point clouds and labels from the batch
        point_clouds, labels = batch
        
        # Select a subset of points based on the 'ind' index and move to the specified device
        point_clouds = point_clouds[:, ind].to(args.device)
        
        # Move labels to the specified device and convert to long type
        labels = labels.to(args.device).to(torch.long)

        # Disable gradient computation during inference
        with torch.no_grad():
            # Get model predictions and find the index with maximum value along the last dimension
            pred_labels = torch.argmax(model(point_clouds), dim=-1, keepdim=False)
        
        # Update the count of correctly predicted objects
        correct_obj += pred_labels.eq(labels.data).cpu().sum().item()
        
        # Update the total number of objects
        num_obj += labels.size()[0]

        # Store the predicted labels for later visualization
        preds_labels.append(pred_labels)

    # Calculate test accuracy based on the collected counts
    accuracy = correct_obj / num_obj

    # Print the test accuracy
    print(f"test accuracy: {accuracy}")

    # Concatenate the predicted labels and move to CPU for further analysis
    preds_labels = torch.cat(preds_labels).detach().cpu()

    # Find indices where predictions do not match ground truth
    fail_inds = torch.nonzero(preds_labels != test_dataloader.dataset.label).squeeze()

    # Visualize a random subset (up to 25 examples) of failed predictions
    for i in range(min(25, len(fail_inds))):
        # Randomly select an index from the predicted labels
        random_ind = random.randint(0, preds_labels.shape[0] - 1)
        
        # Ensure the selected index is among the failed predictions
        while random_ind in fail_inds:
            random_ind = random.randint(0, preds_labels.shape[0] - 1)
        
        # Extract data for visualization
        verts = test_dataloader.dataset.data[random_ind, ind]
        gt_cls = test_dataloader.dataset.label[random_ind].to(torch.long).detach().cpu().data
        pred_cls = preds_labels[random_ind].detach().cpu().data
        
        # Define the file path for visualization
        path = f"output/cls/random_vis_{random_ind}_with_gt_{gt_cls}_pred_{pred_cls}.gif"
        
        # Visualize the object with ground truth and predicted labels and save as GIF files
        viz_cls(verts, path, "cuda", gt_cls, pred_cls)

    # Visualize all failed predictions
    for i in range(len(fail_inds)):
        # Get the index of a failed prediction
        fail_ind = fail_inds[i]
        
        # Extract data for visualization
        verts = test_dataloader.dataset.data[fail_ind, ind]
        gt_cls = test_dataloader.dataset.label[fail_ind].detach().cpu().data
        pred_cls = preds_labels[fail_ind].detach().cpu().data
        
        # Define the file path for visualization
        path = f"output/cls/fail_vis_{fail_ind}_with_gt_{gt_cls}_pred_{pred_cls}.gif"
        
        # Visualize the object with ground truth and predicted labels and save as GIF files
        viz_cls(verts, path, "cuda", gt_cls, pred_cls)