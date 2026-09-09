import torch
import torch.nn as nn
from datasets import DATA_PARAMS


ENCODER_OUT_DIM = 16
LATENT_DIM = 8

HYPERPARAMETERS = {
    "batch_size": 64,
    "learning_rate": 1e-4,
    "validation_split": 0.2,
    "alpha": 0.5

}


"""
Convolutional architecture for encoding the input signals.

Model structure summary.
Input:              [N, 9, 128]
Conv1:              [N, 9, 128] -> [N, 32, 124]
Activation:         no change
Pooling1:           [N, 32, 124] -> [N, 32, 62]
BatchNorm1:         no change
Conv2:              [N, 32, 62] -> [N, 16, 60]
Activation:         no change
Pooling2:           [N, 16, 60] -> [N, 16, 30]
BatchNorm2:         no change
GlobAvgPooling:     [N, 16, 30] -> [N, 16, 1]
Flatten:            [N, 16, 1] -> [N, 16]

(N = batch size)
"""
class SignalEncoder (nn.Module):
    def __init__ (self):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=DATA_PARAMS["num_channels"], out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3)

        self.activation = nn.ReLU()
        self.pooling = nn.MaxPool1d(kernel_size=2)
        self.batchNorm1 = nn.BatchNorm1d(num_features=32, affine=True)
        self.batchNorm2 = nn.BatchNorm1d(num_features=16, affine=True)
        self.globAvgPoolong = nn.AdaptiveAvgPool1d(1)

    def forward (self, x):
        x = self.batchNorm1(self.pooling(self.activation(self.conv1(x))))
        x = self.batchNorm2(self.pooling(self.activation(self.conv2(x))))
        x = torch.flatten(self.globAvgPoolong(x), 1)

        return x


"""
Fully-connected architecture encoding the values of the concepts.

Model structure summary.
Input:              [N, 16]
Dense1:             [N, 16] -> [N, 4]

(N = batch size)
"""
class ConceptHead (nn.Module):
    def __init__ (self):
        super().__init__()

        self.dense1 = nn.Linear(in_features=ENCODER_OUT_DIM, out_features=3)
        self.activation = nn.ReLU()

    def forward (self, x):
        x = self.activation(self.dense1(x))

        return x


"""
Fully-connected architecture performing the final classification task on the activities.

Model structure summary.
Input:              [N, M]
Dense1:             [N, M] -> [N, 8]
Activation:         no change
Dense2:             [N, 8] -> [N, 6]

(N = batch size, M = 16 if concepts=False, M = 4 if concepts=True)
"""
class TaskHead (nn.Module):
    def __init__ (self, concepts=False):
        super().__init__()

        if concepts:
            self.dense1 = nn.Linear(in_features=3, out_features=LATENT_DIM)
        else:
            self.dense1 = nn.Linear(in_features=ENCODER_OUT_DIM, out_features=LATENT_DIM)
        self.dense2 = nn.Linear(in_features=LATENT_DIM, out_features=DATA_PARAMS["num_classes"])

        #self.dropout = nn.Dropout(0.5)
        self.activation = nn.ReLU()


    def forward (self, x):
        x = self.activation(self.dense1(x))
        x = self.dense2(x)

        return x

