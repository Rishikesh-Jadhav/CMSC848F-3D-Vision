import argparse
import time
import torch
from model import SingleViewto3D
from r2n2_custom import R2N2
from  pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
import dataset_location
import pytorch3d
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.ops import knn_points
import mcubes
import utils_vox
import matplotlib.pyplot as plt 
from torchvision import transforms
from matplotlib import pyplot as plt
from PIL import Image


def get_args_parser():
    parser = argparse.ArgumentParser('Singleto3D', add_help=False)
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--max_iter', default=5000, type=int)
    parser.add_argument('--vis_freq', default=1000, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=10000, type=int) 
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=2.0, type=float)  
    parser.add_argument('--load_checkpoint', action='store_true')  
    parser.add_argument('--device', default='cuda', type=str) 
    parser.add_argument('--load_feat', action='store_true') 
    return parser

# Preprocess input images and meshes
def preprocess_images_and_mesh(feed_dict, args):
    for k in ['images']:
        feed_dict[k] = feed_dict[k].to(args.device)

    images = feed_dict['images'].squeeze(1)
    mesh = feed_dict['mesh']
    if args.load_feat:
        images = torch.stack(feed_dict['feats']).to(args.device)

    return images, mesh

# Display and save an image
def display_image(image, output_path):
    plt.imshow(image, interpolation='nearest', cmap='gray')
    plt.savefig(output_path)

# Display and save filter activations
def display_filters(predictions, output_path):
    num_filters = len(predictions)
    rows, cols = 4, 4 

    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))

    for i, ax in enumerate(axes.flatten()):
        if i < num_filters:
            filter_image = predictions[i]
            ax.imshow(filter_image, cmap='viridis')  # Using 'viridis' colormap for an interesting look
            ax.axis("off")
        else:
            ax.axis("off")  

    plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Adjust spacing between subplots
    plt.savefig(output_path)


# Load the test dataset and evaluate the model used from eval_model.py
def evaluate_model(args):
    r2n2_dataset = R2N2("test", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True, return_feats=args.load_feat)

    loader = torch.utils.data.DataLoader(
        r2n2_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=True,
        drop_last=True)
    eval_loader = iter(loader)

    model =  SingleViewto3D(args)
    model.to(args.device)
    model.eval()

    start_iter = 0

    if args.load_checkpoint:
        checkpoint = torch.load(f'checkpoint_{args.type}.pth') 
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Succesfully loaded iter {start_iter}")

    class Interpret_model(torch.nn.Module):
        """
        The model which encodes the normalized input image and normalize it to feed into the decoder.
        """
        def __init__(self):
            super(Interpret_model, self).__init__()
            self.features = torch.nn.Sequential(
                *list(model.encoder.children())[:-2]
            )
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        def forward(self, x):
            B = x.shape[0]
            x_normalize = self.normalize(x.permute(0,3,1,2))
            x = self.features(x_normalize)
            return x
        
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    max_iter = len(eval_loader)
    for step in range(start_iter, max_iter):
        feed_dict = next(eval_loader)

        images_gt, mesh_gt = preprocess_images_and_mesh(feed_dict, args)

        model1 = Interpret_model()
        model1.features[-1].register_forward_hook(get_activation('features'))

        predictions = model1(images_gt)
        conv_layers = []
        model_children = list(model.encoder.children())[:2]
        for child in model_children:
            if type(child)==torch.nn.Conv2d:
                conv_layers.append(child)
            elif type(child)==torch.nn.Sequential:
                for layer in child.children():
                   if type(child)==torch.nn.Conv2d:
                        conv_layers.append(layer) 
    
        img_step = step % 50 
        num = step // 50 
        if img_step == 0:
            img_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])(images_gt.permute(0,3,1,2))
            predictions = [conv_layers[0](img_normalize)]
            for i in range(1, len(conv_layers)):
                predictions.append(conv_layers[i](predictions[-1]))
            display_filters(predictions[-1][0].cpu().detach().numpy(), output_path=f'interpreting_model/Q_2.6-pred-{num}.png') # display the filter activations at the second last layer

    print("Done")      

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Singleto3D', parents=[get_args_parser()])
    args = parser.parse_args()
    evaluate_model(args)
