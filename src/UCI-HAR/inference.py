import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from random import randrange

from models import SignalEncoder, ConceptHead, TaskHead, HYPERPARAMETERS
from datasets import ConceptHARDataset, HARDataset
from concepts import concept_criterion
from metrics import accuracy, plot_conf_matrix


###
# The function performs the inference of the baseline model on the test set and outputs the related metrics.
###
def baseline_inference ():
    test_dataset = HARDataset("test")
    test_dataloader = DataLoader(test_dataset, batch_size=HYPERPARAMETERS["batch_size"], shuffle=False)

    # Model components
    encoder = SignalEncoder()
    task_head = TaskHead()
    model = nn.Sequential(encoder, task_head)

    task_criterion = nn.CrossEntropyLoss()

    # We load the last baseline model saved
    model.load_state_dict(torch.load("src/UCI-HAR/weights/weights_baseline.pt", weights_only=True))

    model.eval()

    avg_loss = 0
    with torch.no_grad():
        for _, data in enumerate(test_dataloader):
            inputs, labels = data

            encoded_signals = encoder(inputs)
            out_labels = task_head(encoded_signals)

            task_loss = task_criterion(out_labels, labels)
            avg_loss += task_loss.item()

        avg_loss = round(avg_loss / len(test_dataloader), 4)
        
        print("Average test loss:", avg_loss)
        print("Task accuracy:", accuracy(model, test_dataloader, concepts=False))
        plot_conf_matrix(model, test_dataloader, title="Confusion matrix related to the test split", concepts=False)


###
# The function performs the inference of the concept bottleneck model on the test set and outputs the related metrics.
###
def concept_inference ():
    test_dataset = ConceptHARDataset("test")
    test_dataloader = DataLoader(test_dataset, batch_size=HYPERPARAMETERS["batch_size"], shuffle=False)

    # Model components
    encoder = SignalEncoder()
    concept_head = ConceptHead()
    task_head = TaskHead(concepts=True)
    model = nn.Sequential(encoder, concept_head, task_head)

    task_criterion = nn.CrossEntropyLoss()

    # We load the last concept bottleneck model saved
    model.load_state_dict(torch.load("src/UCI-HAR/weights/weights_CBM.pt", weights_only=True))

    model.eval()

    avg_loss = 0
    with torch.no_grad():
        for _, data in enumerate(test_dataloader):
            inputs, labels, concepts = data

            encoded_signals = encoder(inputs)
            out_concepts = concept_head(encoded_signals)
            out_labels = task_head(out_concepts)

            concept_loss = concept_criterion(out_concepts, concepts)
            task_loss = task_criterion(out_labels, labels)
            loss = task_loss + HYPERPARAMETERS["alpha"] * concept_loss

            avg_loss += loss.item()

        avg_loss = round(avg_loss / len(test_dataloader), 4)

        print("Average test loss:", avg_loss)
        print("Task accuracy:", accuracy(model, test_dataloader, concepts=True))
        plot_conf_matrix(model, test_dataloader, title="Confusion matrix related to the test split", concepts=True)


###
# The function performs the inference of the concept bottleneck model on a single (random) sample of the test set. The inference
# compares both the predicted concept with the actual concept and the predicted activity with the actual activity
###
def single_concept_inference ():
    test_dataset = ConceptHARDataset("test")
    # We choose a random sample from the test set
    rand_idx = randrange(len(test_dataset))
    rand_sample, label, concepts = test_dataset.__getitem__(rand_idx)
    rand_sample = (torch.from_numpy(rand_sample))[None,:]

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