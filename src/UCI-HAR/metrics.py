import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import DATA_PARAMS


###
# The function calculates the accuracy of the activity classification for a single batch. The returned value is the sum of the
# accuracies of all the samples of the batch.
###
def batch_accuracy (outputs, labels):
    part_acc = 0

    for i in range(0, (outputs.size())[0]):
        y_pred = torch.argmax(outputs[i])

        if y_pred == labels[i]:
            part_acc += 1
        
    return part_acc


###
# The function calculates the accuracy of the activity clasification for a specific dataloader (train/validation/test).
###
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
            # The batch accuracies are accumulated
            acc += batch_accuracy(outputs, labels)

    # The final accuracy is normalized by the number of batches
    return round((acc / len(dataloader.dataset) * 100), 2)


###
# The function calculates the accuracy of the concept prediction for a single batch. The returned value is the sum of the
# correct predictions of all the concepts of all the samples of the batch.
###
def concept_batch_accuracy (outputs, concepts):
    part_acc = 0

    for i in range(0, (outputs.size())[0]):
        for j in range(0, (outputs.size())[1]):
            if round((outputs[i][j]).item()) == int((concepts[i][j]).item()):
                part_acc += 1

    return part_acc


###
# The function plots the confusion matrix and the F1-score table related to a specific dataloader
###
def plot_conf_matrix (model, dataloader, title, concepts=False):
    conf_matrix = np.zeros((DATA_PARAMS["num_classes"], DATA_PARAMS["num_classes"]))

    with torch.no_grad():
        # The data is forwarded into the model in order to fill out the confusion matrix
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

    # Plotting the confusion matrix
    _, ax = plt.subplots()
    ax.xaxis.set_label_position("top")
    ax.yaxis.set_label_position("right")
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="g", xticklabels=labels, yticklabels=labels, cbar=False)
    plt.title(title, fontsize=20, pad=10)
    plt.xlabel("Predicted class", fontsize=14)
    plt.ylabel("Actual class", fontsize=14)
    plt.show()


    precisions = np.zeros((DATA_PARAMS["num_classes"],))
    recalls = np.zeros(precisions.shape)
    f1_scores = np.zeros(precisions.shape)

    # The precision, recall and F1-score is calculated separately for each activity exploiting the confusion matrix
    for idx in range(0, DATA_PARAMS["num_classes"]):
        # Precision(c) = TP(c) / (TP(c) + FP(c))
        precisions[idx] = conf_matrix[idx][idx] / (np.sum(conf_matrix[idx]))
        # Recall(a) = TP(c) / (TP(c) + FN(c))
        recalls[idx] = conf_matrix[idx][idx] / (np.sum(conf_matrix[:, idx]))
        # F1-score(c) = (2 * Precision(c) * Recall(c)) / (Precision(c) + Recall(c))
        f1_scores[idx] = (2 * precisions[idx] * recalls[idx]) / (precisions[idx] + recalls[idx])

    # The obtained metrics are merged to be plotted
    table_data = np.stack((precisions, recalls, f1_scores), axis=1) 
    col_labels = ["Precision", "Recall", "F1-score"]

    # Plotting the table containing precision, recall and F1-score for each activity 
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')

    table = ax.table(cellText=np.round(table_data, 2), colLabels=col_labels, rowLabels=labels, loc="center", cellLoc="center")
    table.scale(1, 1.5)
    table.auto_set_font_size(False)
    table.set_fontsize(12)

    fig.tight_layout()
    plt.show()


###
# The function plots the activity classification accuracy separated by subject, given a specific dictionary of statistics that
# associates each subject to the pair (number of correct predictions for that subject, number of instances of the subject)
###
def plot_accuracy_per_subject (subjects_stats):
    acc_per_subj = np.zeros((len(subjects_stats.keys()),))
    idx = 0
    for subj_idx in subjects_stats:
        # Accuracy = (number of correct predictions for that subject) / (number of instances of the subject)
        acc_per_subj[idx] = round(subjects_stats[subj_idx][0] / subjects_stats[subj_idx][1], 2)
        idx += 1

    x = list(map(str, list(subjects_stats.keys())))
    plt.figure(figsize=(10, 5))
    bars = plt.bar(x, acc_per_subj)
    plt.bar_label(bars)

    plt.title("Task accuracy per subject", fontsize=20, pad=10)
    plt.xlabel("Subject", fontsize=15)
    plt.ylabel("Accuracy", fontsize=15)
    plt.show()