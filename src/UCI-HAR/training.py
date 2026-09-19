import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt 

from datasets import HARDataset, ConceptHARDataset, split_for_validation
from models import HYPERPARAMETERS, SignalEncoder, TaskHead, ConceptHead
from metrics import accuracy, plot_conf_matrix, concept_batch_accuracy
from concepts import CONCEPTS_LAYER_DIM, concept_criterion


NUM_EPOCHS = 100
PATIENCE = 10


###
# The function performs the training of the baseline model.
###
def baseline_train ():
    train_dataset = HARDataset("train")
    train_subset, validation_subset = split_for_validation(train_dataset)

    train_dataloader = DataLoader(train_subset, batch_size=HYPERPARAMETERS["batch_size"], shuffle=True)
    validation_dataloader = DataLoader(validation_subset, batch_size=HYPERPARAMETERS["batch_size"], shuffle=False)

    # Model components
    encoder = SignalEncoder()
    taskHead = TaskHead()
    model = nn.Sequential(encoder, taskHead)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=HYPERPARAMETERS["learning_rate"])

    # Train and validation accuracy (w.r.t. the activity) associated to the model before the training phase
    print("Starting point")
    print("Train accuracy: {0}%".format(accuracy(model, train_dataloader)))
    print("Validation accuracy: {0}%\n".format(accuracy(model, validation_dataloader)))

    y_train_loss = []
    y_val_loss = []
    y_train_acc = []
    y_val_acc = []

    best_val_loss = float("inf")
    prev_val_loss = float("inf")
    best_epoch = 0
    final_val_acc = 0
    best_model = None
    epochs = 0
    current_patience = 0

    # Training loop
    while (True):
        epochs += 1
        print("Current epoch:", epochs, end="\r")

        model.train()
    
        train_loss = 0
        # Iterating through the training batches
        for _, data in enumerate(train_dataloader):
            inputs, labels = data
            
            optimizer.zero_grad()
    
            # The current batch is forwarded to the network
            outputs = model(inputs)

            # Loss calculation
            loss = criterion(outputs, labels)
            train_loss += loss.item()
    
            # Backpropagation 
            loss.backward()
            # Parameters update
            optimizer.step()
    
        # The model is set in evaluation mode
        model.eval()
        val_loss = 0

        with torch.no_grad():
            # Iterating through the validation batches
            for _, data in enumerate(validation_dataloader):
                inputs, labels = data
                outputs = model(inputs)
    
                loss = criterion(outputs, labels)
                val_loss += loss.item()
    
        # Epoch loss calculation
        train_loss = round(train_loss / len(train_dataloader), 4)
        val_loss = round(val_loss / len(validation_dataloader), 4)
        y_train_loss.append(train_loss)
        y_val_loss.append(val_loss)
    
        # Epoch accuracy calculation
        train_acc = accuracy(model, train_dataloader)
        val_acc = accuracy(model, validation_dataloader)
        y_train_acc.append(train_acc)
        y_val_acc.append(val_acc)
    
        # Best model update
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            final_val_acc = val_acc
            best_epoch = epochs
            best_model = model.state_dict()

        # Early stopping verification
        if val_loss > prev_val_loss:
            current_patience += 1
            if current_patience == PATIENCE:
                break
        else:
            current_patience = 0
            prev_val_loss = val_loss


    x = np.linspace(1, epochs, epochs)
    # Plotting the loss variation of the training and validation splits
    plt.plot(x, y_train_loss, label="Train loss")
    plt.plot(x, y_val_loss, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss variation for train/validation splits")
    plt.legend()
    plt.axvline(x=best_epoch, linestyle="dashed")
    plt.show()

    # Plotting the accuracy variation of the training and validation splits
    plt.plot(x, y_train_acc, label="Train accuracy")
    plt.plot(x, y_val_acc, label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy variation for train/validation splits")
    plt.legend()
    plt.axvline(x=best_epoch, linestyle="dashed")
    plt.show()

    print("\nBest validation loss:", best_val_loss)
    print("Final validation accuracy: ", str(final_val_acc) + "%")

    # Plotting the confusion matrices and the F1-score tables for train/validation splits
    plot_conf_matrix(model, train_dataloader, title="Confusion matrix related to the train split")
    plot_conf_matrix(model, validation_dataloader, title="Confusion matrix related to the validation split")

    return best_model


###
# The function performs the training of the concept bottleneck model.
###
def concept_train ():
    train_dataset = ConceptHARDataset("train")
    train_split, validation_split = split_for_validation(train_dataset)

    train_dataloader = DataLoader(train_split, batch_size=HYPERPARAMETERS["batch_size"], shuffle=True)
    validation_dataloader = DataLoader(validation_split, batch_size=HYPERPARAMETERS["batch_size"], shuffle=False)

    # Model components
    encoder = SignalEncoder()
    conceptHead = ConceptHead()
    taskHead = TaskHead(concepts=True)
    model = nn.Sequential(encoder, conceptHead, taskHead)

    task_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=HYPERPARAMETERS["learning_rate"])

    # Train and validation accuracy (w.r.t. the activity) associated to the model before the training phase
    print("Starting point")
    print("Train accuracy: {0}%".format(accuracy(model, train_dataloader, concepts=True)))
    print("Validation accuracy: {0}%\n".format(accuracy(model, validation_dataloader, concepts=True)))

    y_train_loss = []
    y_val_loss = []
    y_train_acc = []
    y_val_acc = []

    y_conc_train_acc = []
    y_conc_val_acc = []

    best_val_loss = float("inf")
    prev_val_loss = float("inf")
    epochs = 0
    current_patience = 0
    best_epoch = 0
    final_val_acc = 0
    best_model = None

    # Training loop
    while (True):
        epochs += 1
        print("Current epoch:", epochs, end="\r")

        model.train()
    
        train_loss = 0
        conc_train_acc = 0
        conc_val_acc = 0
        # Iterating through the training batches
        for _, data in enumerate(train_dataloader):
            inputs, labels, concepts = data
            
            optimizer.zero_grad()

            # The current batch is forwarded to the network
            encoded_input = encoder(inputs)
            out_concepts = conceptHead(encoded_input)
            out_labels = taskHead(out_concepts)

            # Loss calculation
            concept_loss = concept_criterion(out_concepts, concepts)
            task_loss = task_criterion(out_labels, labels)
            loss = task_loss + HYPERPARAMETERS["alpha"] * concept_loss
            train_loss += loss.item()
    
            # Backpropagation 
            loss.backward()
            # Parameters update
            optimizer.step()

            conc_train_acc += concept_batch_accuracy(out_concepts, concepts)
    
        # The model is set in evaluation mode
        model.eval()
        val_loss = 0

        with torch.no_grad():
            # Iterating through the validation batches
            for _, data in enumerate(validation_dataloader):
                inputs, labels, concepts = data

                encoded_input = encoder(inputs)
                out_concepts = conceptHead(encoded_input)
                out_labels = taskHead(out_concepts)
    
                concept_loss = concept_criterion(out_concepts, concepts)
                task_loss = task_criterion(out_labels, labels)
                loss = task_loss + HYPERPARAMETERS["alpha"] * concept_loss
                val_loss += loss.item()

                conc_val_acc += concept_batch_accuracy(out_concepts, concepts)

    
        # Epoch loss calculation
        train_loss = round(train_loss / len(train_dataloader), 4)
        val_loss = round(val_loss / len(validation_dataloader), 4)
        y_train_loss.append(train_loss)
        y_val_loss.append(val_loss)
    
        # Epoch accuracy calculation (w.r.t. the activity)
        train_acc = accuracy(model, train_dataloader, concepts=True)
        val_acc = accuracy(model, validation_dataloader, concepts=True)
        y_train_acc.append(train_acc)
        y_val_acc.append(val_acc)

        # Epoch accuracy calculation (w.r.t. the concepts)
        conc_train_acc = round((conc_train_acc / (len(train_dataloader.dataset) * CONCEPTS_LAYER_DIM) * 100), 2)
        y_conc_train_acc.append(conc_train_acc)
        conc_val_acc = round((conc_val_acc / (len(validation_dataloader.dataset) * CONCEPTS_LAYER_DIM) * 100), 2)
        y_conc_val_acc.append(conc_val_acc)
    
        # Best model update
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            final_val_acc = val_acc
            best_epoch = epochs
            best_model = model.state_dict()

        # Early stopping verification 
        if val_loss > prev_val_loss:
            current_patience += 1
            if current_patience == PATIENCE:
                break
        else:
            current_patience = 0
            prev_val_loss = val_loss


    x = np.linspace(1, epochs, epochs)
    # Plotting the loss variation of the training and validation splits
    plt.plot(x, y_train_loss, label="Train loss")
    plt.plot(x, y_val_loss, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss variation for train/validation splits")
    plt.legend()
    plt.axvline(x=best_epoch, linestyle="dashed")
    plt.show()

    # Plotting the accuracy variation of the training and validation splits both w.r.t. the activities and the concepts
    plt.plot(x, y_train_acc, label="Train accuracy", color="blue")
    plt.plot(x, y_val_acc, label="Validation accuracy", color="orange")
    plt.plot(x, y_conc_train_acc, label="Concept train accuracy", color="blue", linestyle="dashed")
    plt.plot(x, y_conc_val_acc, label="Concept validation accuracy", color="orange", linestyle="dashed")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy variation for train/validation splits")
    plt.legend()
    plt.axvline(x=best_epoch, linestyle="dashed")
    plt.show()

    print("\nBest validation loss:", best_val_loss)
    print("Final validation accuracy: ", str(final_val_acc) + "%")

    # Plotting the confusion matrices and the F1-score tables for train/validation splits
    plot_conf_matrix(model, train_dataloader, title="Confusion matrix related to the train split", concepts=True)
    plot_conf_matrix(model, validation_dataloader, title="Confusion matrix related to the validation split", concepts=True)

    return best_model


###
# The function saves the model weights produced by a training process.
###
def save_model(best_model, concepts=False):
    if best_model is not None:
        if concepts:
            torch.save(best_model, "src/UCI-HAR/weights/weights_CBM.pt")
        else:
            torch.save(best_model, "src/UCI-HAR/weights/weights_baseline.pt")
    else:
        print("No best model!")



if __name__ == "__main__":
    best_model = concept_train()
    save_model(best_model, concepts=True)