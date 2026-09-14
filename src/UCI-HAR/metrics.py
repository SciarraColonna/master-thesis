import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import DATA_PARAMS


"""
The function calculates the accuracy of the activity classification for a single batch. The returned value is the sum of the
accuracies of all the samples of the batch.
"""
def batch_accuracy (outputs, labels):
    part_acc = 0

    for i in range(0, (outputs.size())[0]):
        y_pred = torch.argmax(outputs[i])

        if y_pred == labels[i]:
            part_acc += 1
        
    return part_acc


"""
The function calculates the accuracy of the activity clasification for a specific dataloader (train/validation/test).
"""
def accuracy (model, dataloader, concepts=False):
    acc = 0
    model.eval()

    with torch.no_grad():
        for _, data in enumerate(dataloader):
            if concepts:
                inputs, labels, _ = data
            else:
                inputs, labels = data
            outputs = model(inputs)
            acc += batch_accuracy(outputs, labels)

    return round((acc / len(dataloader.dataset) * 100), 2)


"""
The function calculates the accuracy of the activity classification for a single batch. The returned value is the sum of the
accuracies of all the samples of the batch.
"""
def concept_batch_accuracy (outputs, concepts):
    part_acc = 0

    for i in range(0, (outputs.size())[0]):
        for j in range(0, (outputs.size())[1]):
            if round((outputs[i][j]).item()) == int((concepts[i][j]).item()):
                part_acc += 1

    return part_acc


"""
The function plots the confusion matrix related to a specific dataloader (train/validation/split).
"""
def plot_conf_matrix (model, dataloader, title, concepts=False):
    conf_matrix = np.zeros((DATA_PARAMS["num_classes"], DATA_PARAMS["num_classes"]))

    with torch.no_grad():
        for _, data in enumerate(dataloader):
            if concepts:
                inputs, labels, _ = data
            else:
                inputs, labels = data
            outputs = model(inputs)

            for i in range(0, (outputs.size())[0]):
                y_pred = torch.argmax(outputs[i])
                y_actual = labels[i]

                conf_matrix[y_actual][y_pred] += 1

    labels = ["WALKING", "WALKING UPSTAIRS", "WALKING DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]

    _, ax = plt.subplots()
    ax.xaxis.set_label_position("top")
    ax.yaxis.set_label_position("right")
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="g", xticklabels=labels, yticklabels=labels, cbar=False)
    plt.title(title, fontsize=20, pad=10)
    plt.xlabel("Predicted class", fontsize=14)
    plt.ylabel("Actual class", fontsize=14)
    plt.show()