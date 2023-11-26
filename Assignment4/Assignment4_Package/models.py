import torch
import torch.nn as nn
import torch.nn.functional as F

# ------ TO DO ------
class cls_model(nn.Module):
    def __init__(self, num_classes=3):
        '''
        Initialize the classification model.

        Args:
        - num_classes (int): The number of output classes.

        Model Architecture:
        - The model consists of a series of convolutional layers followed by fully connected layers.
        - Each convolutional layer is followed by batch normalization and ReLU activation.
        - The last fully connected layer produces the final class predictions.

        Note:
        - The input tensor is expected to have a shape of (B, 3, N), where B is the batch size and N is the number of points per object.
        - The default value for N is 10000.
        '''        
        super(cls_model, self).__init__()

        # Define convolutional layers
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.conv4 = nn.Conv1d(128, 1024, 1)

        # Define batch normalization layers
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(1024)

        # Define fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )
        pass

    def forward(self, points):
        '''
        Forward pass of the model.

        Args:
        - points (Tensor): Input tensor of shape (B, 3, N), where B is batch size and N  (N=10000 by default) is the number of points per object.

        Returns:
        - output (Tensor): Output tensor of shape (B, num_classes).

        Model Operation:
        - Transpose the input tensor to the shape (B, N, 3).
        - Apply a series of convolutional layers with batch normalization and ReLU activation.
        - Perform max-pooling operation along the last dimension.
        - Pass the resulting feature through fully connected layers to produce class predictions.
        '''
        # Transpose input tensor to (B, N, 3)
        points = points.transpose(1, 2)

        # Apply convolutional layers with batch normalization and ReLU
        out = F.relu(self.bn1(self.conv1(points)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        out = F.relu(self.bn4(self.conv4(out)))

        # Max-pooling operation
        out, _ = torch.max(out, dim=-1)

        # Pass the feature through fully connected layers
        out = self.fc(out)

        return out




# ------ TO DO ------
class seg_model(nn.Module):
    def __init__(self, num_seg_classes = 6):
        '''
        Initialize the segmentation model.

        Args:
        - num_seg_classes (int): The number of segmentation classes.

        Model Architecture:
        - The model consists of a series of convolutional layers for feature extraction.
        - Two sets of convolutional layers are used, one for local features and the other for global features.
        - Each convolutional layer is followed by batch normalization and ReLU activation.
        - The local and global features are combined and passed through additional convolutional layers for segmentation.

        Note:
        - The input tensor is expected to have a shape of (B, 3, N), where B is the batch size and N is the number of points per object.
        - The default value for N is 10,000.
        '''
        super(seg_model, self).__init__()

        # Define convolutional layers for local features
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)

        # Define convolutional layers for global features
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.conv4 = nn.Conv1d(128, 1024, 1)

        # Batch normalization layers for local and global features
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(1024)

        # Point layer for segmentation
        self.point_layer = nn.Sequential(
            nn.Conv1d(1088, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, num_seg_classes, 1),
        )
        pass

    def forward(self, points):
        '''
        Forward pass of the model.

        Args:
        - points (Tensor): Input tensor of shape (B, N, 3), where B is batch size, N (N=10000 by default) is the number of points per object.

        Returns:
        - output (Tensor): Output tensor of shape (B, N, num_seg_classes).

        Model Operation:
        - Transpose the input tensor to the shape (B, 3, N).
        - Apply convolutional layers with batch normalization and ReLU for local features.
        - Apply convolutional layers with batch normalization and ReLU for global features.
        - Perform max-pooling to obtain a global feature representation.
        - Concatenate local and global features.
        - Pass the concatenated feature through convolutional layers for segmentation.
        '''
        N = points.shape[1]
        points = points.transpose(1, 2)

        # Local feature extraction
        local_out = F.relu(self.bn1(self.conv1(points)))
        local_out = F.relu(self.bn2(self.conv2(local_out)))

        # Global feature extraction
        global_out = F.relu(self.bn3(self.conv3(local_out)))
        global_out = F.relu(self.bn4(self.conv4(global_out)))
        global_out = torch.amax(global_out, dim=-1, keepdims=True).repeat(1, 1, N)

        # Combine local and global features
        out = torch.cat((local_out, global_out), dim=1)

        # Pass through convolutional layers for segmentation
        out = self.point_layer(out).transpose(1, 2)

        return out




