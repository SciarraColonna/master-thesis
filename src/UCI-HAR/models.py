import torch
import torch.nn as nn


class BaselineModel (nn.Module):
    def __init__ (self, out_classes):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=NUM_CHANNELS, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3)
        self.pooling = nn.MaxPool1d(kernel_size=2)
        self.globAvgPooling = nn.AdaptiveAvgPool1d(1)
        self.dense = nn.Linear(in_features=16, out_features=out_classes)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.batchNorm1 = nn.BatchNorm1d(num_features=32, affine=True)
        self.batchNorm2 = nn.BatchNorm1d(num_features=16, affine=True)

    def forward (self, x):
        x = self.batchNorm1(self.pooling(self.activation(self.conv1(x))))
        x = self.batchNorm2(self.pooling(self.activation(self.conv2(x))))
        x = self.globAvgPooling(x)
        x = self.activation(torch.flatten(x, 1))
        x = self.dense(x)
        
        return x


"""model = BaselineModel(NUM_CLASSES)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)"""