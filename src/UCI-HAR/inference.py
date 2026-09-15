import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from random import randrange

from models import SignalEncoder, ConceptHead, TaskHead, HYPERPARAMETERS
from datasets import ConceptHARDataset, HARDataset
from concepts import concept_criterion
from metrics import accuracy, plot_conf_matrix


def baseline_inference ():
    test_dataset = HARDataset("test")
    test_dataloader = DataLoader(test_dataset, batch_size=HYPERPARAMETERS["batch_size"], shuffle=False)

    encoder = SignalEncoder()
    task_head = TaskHead()

    task_criterion = nn.CrossEntropyLoss()

    model = nn.Sequential(encoder, task_head)
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
        
        print(avg_loss)
        print(accuracy(model, test_dataloader, concepts=False))
        plot_conf_matrix(model, test_dataloader, title="Confusion matrix related to the test split", concepts=False)



def concept_inference ():
    test_dataset = ConceptHARDataset("test")
    test_dataloader = DataLoader(test_dataset, batch_size=HYPERPARAMETERS["batch_size"], shuffle=False)

    encoder = SignalEncoder()
    concept_head = ConceptHead()
    task_head = TaskHead(concepts=True)

    task_criterion = nn.CrossEntropyLoss()

    model = nn.Sequential(encoder, concept_head, task_head)
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

        print(avg_loss)
        print(accuracy(model, test_dataloader, concepts=True))
        plot_conf_matrix(model, test_dataloader, title="Confusion matrix related to the test split", concepts=True)


def single_concept_inference ():
    test_dataset = ConceptHARDataset("test")
    rand_idx = randrange(len(test_dataset))
    rand_sample, label, concepts = test_dataset.__getitem__(rand_idx)
    rand_sample = (torch.from_numpy(rand_sample))[None,:]

    encoder = SignalEncoder()
    concept_head = ConceptHead()
    task_head = TaskHead(concepts=True)

    model = nn.Sequential(encoder, concept_head, task_head)
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
    concept_inference()