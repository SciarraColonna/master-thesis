import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset

import numpy as np
import math



DATA_PARAMS = {
    "num_classes": 6,
    "num_channels": 9,
    "num_subjects": 30,
    "train_subjects": 21,
    "validation_split": 0.2,

    "activity_labels": {
        0: "WALKING",
        1: "WALKING_UPSTAIRS",
        2: "WALKING_DOWNSTAIRS",
        3: "SITTING",
        4: "STANDING",
        5: "LAYING"
    }
}

data_path = {
    "train": "../data/UCI-HAR/train/",
    "test": "../data/UCI-HAR/test/"
}

signals_path = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z"
]


"""
This function splits the train Dataset object into a proper train split and a validation split, divided according to a specific 
percentage of the train subjects.
The "validation_split" parameter indicates the percentage of the training subjects (21) that must be included in the validation
split. In particular, the first (1-p)*100 % of the train subjects are used for the training split and the remaining p*100 % are
used for the validation split.
The function produces two Subset objects.
"""
def split_for_validation (train_dataset):
    # Number of subjects for the validation split
    val_subjects_split = math.floor(DATA_PARAMS["validation_split"] * DATA_PARAMS["train_subjects"])
    # Number of subjects for the train split
    train_subjects_split = DATA_PARAMS["train_subjects"] - val_subjects_split

    train_subjects = np.loadtxt(data_path["train"] + "subject_train.txt").astype("int")

    # Calculation of the first sample belonging to the validation set
    counter = 0
    curr_subj = 0
    val_split_start = 0
    for subj in train_subjects:
        val_split_start += 1
        if subj != curr_subj:
            curr_subj = subj
            counter += 1
        if counter == train_subjects_split + 1:
            break

    # The train Dataset object is split into two Subset objects
    train_subset = Subset(train_dataset, [i for i in range(0, val_split_start)])
    validation_subset = Subset(train_dataset, [i for i in range (val_split_start, train_dataset.__len__())])

    return (train_subset, validation_subset)


"""
Dataset class corresponding to the UCI-HAR dataset without the concepts.
For each row, the class returns the corresponding data sample and the related activity label.
"""
class HARDataset (Dataset):
    def __init__ (self, type):
        # Activity labels
        self.labels = np.loadtxt(data_path[type] + "y_" + type + ".txt").astype("int")
        # Dataset subjects
        self.subjects = np.loadtxt(data_path[type] + "subject_" + type + ".txt").astype("int")

        # Raw signals data
        for idx in range(0, len(signals_path)):
            signals = np.loadtxt(data_path[type] + "Inertial Signals/" + signals_path[idx] + "_" + type + ".txt").astype("float32")
            if idx > 0:
                self.data = np.dstack((self.data, signals))
            else:
                self.data = signals

    def __len__ (self):
        return len(self.labels)

    def __getitem__ (self, idx):
        # sample: [128, 9]
        sample = self.data[idx, :, :]
        # sample: [9, 128] -> the channel dimension is the first being the depth of the input data
        sample = np.transpose(sample)
        label = torch.tensor(self.labels[idx] - 1, dtype=torch.long)

        return sample, label


"""
Dataset class corresponding to the UCI-HAR dataset with the concepts.
For each row, the class returns the corresponding data sample and the related activity label together with the concepts values.
"""
class ConceptHARDataset (Dataset):
    def __init__ (self, type):
        # Activity labels
        self.labels = np.loadtxt(data_path[type] + "y_" + type + ".txt").astype("int")
        # Dataset subjects
        self.subjects = np.loadtxt(data_path[type] + "subject_" + type + ".txt").astype("int")
        # Dataset concepts
        self.concepts = np.loadtxt(data_path[type] + "concepts_" + type + ".txt").astype("float32")

        # Raw signals data
        for idx in range(0, len(signals_path)):
            signals = np.loadtxt(data_path[type] + "Inertial Signals/" + signals_path[idx] + "_" + type + ".txt").astype("float32")
            if idx > 0:
                self.data = np.dstack((self.data, signals))
            else:
                self.data = signals

    def __len__ (self):
        return len(self.labels)

    def __getitem__ (self, idx):
        # sample: [128, 9]
        sample = self.data[idx, :, :]
        # sample: [9, 128] -> the channel dimension is the first being the depth of the input data
        sample = np.transpose(sample)
        label = torch.tensor(self.labels[idx] - 1, dtype=torch.long)
        item_concepts = torch.from_numpy(self.concepts[idx])

        return sample, label, item_concepts
