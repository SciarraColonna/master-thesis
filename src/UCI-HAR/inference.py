import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import SignalEncoder, ConceptHead, TaskHead, HYPERPARAMETERS
from datasets import ConceptHARDataset
from concepts import concept_criterion
from metrics import accuracy


def concept_inference ():
    test_dataset = ConceptHARDataset("test")
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    encoder = SignalEncoder()
    concept_head = ConceptHead()
    task_head = TaskHead(concepts=True)

    task_criterion = nn.CrossEntropyLoss()
    #concept_criterion = nn.BCELoss()

    model = nn.Sequential(encoder, concept_head, task_head)
    model.load_state_dict(torch.load("model/weights.pt", weights_only=True))

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

        avg_loss = round(avg_loss / len(test_dataloader.dataset), 4)

        print(avg_loss)
        print(accuracy(model, test_dataloader, concepts=True))


if __name__ == "__main__":
    concept_inference()