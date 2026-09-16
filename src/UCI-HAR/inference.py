import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from random import randrange
import matplotlib.pyplot as plt

from models import SignalEncoder, ConceptHead, TaskHead, HYPERPARAMETERS
from datasets import ConceptHARDataset, HARDataset
from concepts import concept_criterion
from metrics import accuracy, batch_accuracy, plot_conf_matrix, plot_accuracy_per_subject


###
# The function performs the inference of the baseline model on the test set and outputs the related metrics.
###
def baseline_inference ():
    test_dataset = HARDataset("test")
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Model components
    encoder = SignalEncoder()
    task_head = TaskHead()
    model = nn.Sequential(encoder, task_head)

    task_criterion = nn.CrossEntropyLoss()

    # We load the last baseline model saved
    model.load_state_dict(torch.load("src/UCI-HAR/weights/weights_baseline.pt", weights_only=True))

    model.eval()

    # Dictionary containing information about the model performance for each subject
    subjects_stats = dict()
    avg_loss = 0
    with torch.no_grad():
        # Iterating through the test batches (batch size = 1)
        for idx, data in enumerate(test_dataloader):
            inputs, labels = data

            encoded_signals = encoder(inputs)
            out_labels = task_head(encoded_signals)

            task_loss = task_criterion(out_labels, labels)
            avg_loss += task_loss.item()

            # The performance-per-subject dictionary is updated
            subj_idx = int(test_dataset.subjects[idx])
            if subj_idx not in subjects_stats:
                subjects_stats[subj_idx] = [0, 0]
            
            subjects_stats[subj_idx][0] += batch_accuracy(out_labels, labels)
            subjects_stats[subj_idx][1] += 1


        avg_loss = round(avg_loss / len(test_dataloader), 4)
        
        print("Average test loss:", avg_loss)
        print("Task accuracy:", accuracy(model, test_dataloader, concepts=False))
        plot_conf_matrix(model, test_dataloader, title="Confusion matrix related to the test split", concepts=False)
        plot_accuracy_per_subject(subjects_stats)


###
# The function performs the inference of the concept bottleneck model on the test set and outputs the related metrics.
###
def concept_inference ():
    test_dataset = ConceptHARDataset("test")
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Model components
    encoder = SignalEncoder()
    concept_head = ConceptHead()
    task_head = TaskHead(concepts=True)
    model = nn.Sequential(encoder, concept_head, task_head)

    task_criterion = nn.CrossEntropyLoss()

    # We load the last concept bottleneck model saved
    model.load_state_dict(torch.load("src/UCI-HAR/weights/weights_CBM.pt", weights_only=True))

    model.eval()

    # Dictionary containing information about the model performance for each subject
    subjects_stats = dict()
    avg_loss = 0
    with torch.no_grad():
        # Iterating through the test batches (batch size = 1)
        for idx, data in enumerate(test_dataloader):
            inputs, labels, concepts = data

            encoded_signals = encoder(inputs)
            out_concepts = concept_head(encoded_signals)
            out_labels = task_head(out_concepts)

            concept_loss = concept_criterion(out_concepts, concepts)
            task_loss = task_criterion(out_labels, labels)
            loss = task_loss + HYPERPARAMETERS["alpha"] * concept_loss

            avg_loss += loss.item()

            # The performance-per-subject dictionary is updated
            subj_idx = int(test_dataset.subjects[idx])
            if subj_idx not in subjects_stats:
                subjects_stats[subj_idx] = [0, 0]
            
            subjects_stats[subj_idx][0] += batch_accuracy(out_labels, labels)
            subjects_stats[subj_idx][1] += 1


        avg_loss = round(avg_loss / len(test_dataloader), 4)

        print("Average test loss:", avg_loss)
        print("Task accuracy:", accuracy(model, test_dataloader, concepts=True))
        plot_conf_matrix(model, test_dataloader, title="Confusion matrix related to the test split", concepts=True)
        plot_accuracy_per_subject(subjects_stats)


###
# The function plots the body acceleration, angular velocity and total acceleration (on the X, Y and Z axes) of a specific sample
# of the test set.
###
def plot_test_sample (sample):
    x = np.linspace(0, 2.56, 128)

    for i in range(0, 9, 3):
        if i == 0: plt.title("Body acceleration (g)")
        if i == 3: plt.title("Angular velocity (rad/s)")
        if i == 6: plt.title("Total body acceleration (g)")

        plt.plot(x, sample[i], label="X axis")
        plt.plot(x, sample[i + 1], label="Y axis")
        plt.plot(x, sample[i + 2], label="Z axis")
        plt.legend()
        plt.ylim((-3, 3))
        plt.show()


###
# The function performs the inference of the concept bottleneck model on a single (random) sample of the test set. The inference
# compares both the predicted concept with the actual concept and the predicted activity with the actual activity.
###
def single_concept_inference ():
    test_dataset = ConceptHARDataset("test")
    # We choose a random sample from the test set
    rand_idx = randrange(len(test_dataset))
    rand_sample, label, concepts = test_dataset.__getitem__(rand_idx)

    plot_test_sample(rand_sample)
    rand_sample = (torch.from_numpy(rand_sample))[None,:]

    # Model components
    encoder = SignalEncoder()
    concept_head = ConceptHead()
    task_head = TaskHead(concepts=True)

    model = nn.Sequential(encoder, concept_head, task_head)
    # We load the last concept bottleneck model saved
    model.load_state_dict(torch.load("src/UCI-HAR/weights/weights_CBM.pt", weights_only=True))
    
    model.eval()

    encoded_signals = encoder(rand_sample)
    out_concepts = concept_head(encoded_signals)
    out_labels = task_head(out_concepts)

    pred_concepts = out_concepts.tolist()
    pred_labels = torch.softmax(out_labels, dim=1).tolist()

    print("Predicted concepts:", [round(val, 2) for val in pred_concepts[0]])
    print("Predicted label:", [round(val, 2) for val in pred_labels[0]])
    print("\n")
    print("Actual concepts:", concepts)
    print("Actual label:", label.item())



if __name__ == "__main__":
    baseline_inference()