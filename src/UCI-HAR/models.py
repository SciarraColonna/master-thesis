import torch
import torch.nn as nn

from datasets import DATA_PARAMS
from concepts import CONCEPTS_LAYER_DIM, CONCEPTS_MAP


ENCODER_OUT_DIM = 32
LATENT_DIM = 8

# Common training hyperparameters
HYPERPARAMETERS = {
    "batch_size": 64,
    "learning_rate": 5e-4,
    "validation_split": 0.2,
    "alpha": 1.0
}


###
# Convolutional architecture for encoding the input signals.

# Model structure summary.
# Input:              [N, 9, 128]
# Conv1:              [N, 9, 128] -> [N, 32, 120]
# Activation:         no change
# Pooling1:           [N, 32, 120] -> [N, 32, 60]
# BatchNorm1:         no change
# Conv2:              [N, 32, 62] -> [N, 64, 56]
# Activation:         no change
# Pooling2:           [N, 64, 60] -> [N, 64, 28]
# BatchNorm2:         no change
# Conv3:              [N, 64, 62] -> [N, 32, 14]
# Activation:         no change
# Pooling3:           [N, 32, 14] -> [N, 32, 7]
# BatchNorm2:         no change
# GlobAvgPooling:     [N, 32, 30] -> [N, 32, 1]
# Flatten:            [N, 32, 1] -> [N, 32]

# (N = batch size)
###
class SignalEncoder (nn.Module):
    def __init__ (self):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=DATA_PARAMS["num_channels"], out_channels=32, kernel_size=9)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3)

        self.activation = nn.ReLU()
        self.pooling = nn.MaxPool1d(kernel_size=2)
        self.batchNorm1 = nn.BatchNorm1d(num_features=32, affine=True)
        self.batchNorm2 = nn.BatchNorm1d(num_features=64, affine=True)
        self.batchNorm3 = nn.BatchNorm1d(num_features=32, affine=True)
        self.globAvgPooling = nn.AdaptiveAvgPool1d(1)

    def forward (self, x):
        x = self.batchNorm1(self.pooling(self.activation(self.conv1(x))))
        x = self.batchNorm2(self.pooling(self.activation(self.conv2(x))))
        x = self.batchNorm3(self.pooling(self.activation(self.conv3(x))))
        x = torch.flatten(self.globAvgPooling(x), 1)

        return x


###
# Fully-connected architecture encoding the values of the concepts.

# Model structure summary.
# Input:              [N, 16]
# Dense1:             [N, 16] -> [N, 4]
# Activation:         no change

# (N = batch size)
###
class ConceptHead (nn.Module):
    def __init__ (self):
        super().__init__()

        self.dense1 = nn.Linear(in_features=ENCODER_OUT_DIM, out_features=CONCEPTS_LAYER_DIM)

    def forward (self, x):
        x = self.dense1(x)

        # The final activation function mixes the sigmoid (for the units that are associated to a binary concept) and the 
        # softmax (for the units that are associated to a ternary concept).
        for i in range(0, (x.size())[0]):
            c_idx = 0
            for j in CONCEPTS_MAP:
                if j == 1:
                    x[i][c_idx] = (x[i][c_idx]).sigmoid()
                else:
                    x[i, c_idx:(c_idx + j)] = torch.softmax(x[i, c_idx:(c_idx + j)], dim=0)
                c_idx += j

        return x


###
# Fully-connected architecture performing the final classification task on the activities.

# Model structure summary.
# Input:              [N, M]
# Dense1:             [N, M] -> [N, 8]
# Activation:         no change
# Dense2:             [N, 8] -> [N, 6]

# (N = batch size, M = ENCODER_OUT_DIM if concepts=False, M = CONCEPTS_LAYER_DIM if concepts=True)
###
class TaskHead (nn.Module):
    def __init__ (self, concepts=False):
        super().__init__()

        if concepts:
            self.dense1 = nn.Linear(in_features=CONCEPTS_LAYER_DIM, out_features=LATENT_DIM)
        else:
            self.dense1 = nn.Linear(in_features=ENCODER_OUT_DIM, out_features=LATENT_DIM)
        self.dense2 = nn.Linear(in_features=LATENT_DIM, out_features=DATA_PARAMS["num_classes"])

        self.activation = nn.ReLU()

    def forward (self, x):
        x = self.activation(self.dense1(x))
        x = self.dense2(x)

        return x

