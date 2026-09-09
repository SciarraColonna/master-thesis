import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm

from datasets import HARDataset, ConceptHARDataset, split_for_validation
from models import HYPERPARAMETERS, SignalEncoder, TaskHead
from metrics import accuracy, plot_conf_matrix


NUM_EPOCHS = 100


def baseline_train ():
    train_dataset = HARDataset("train")
    #test_dataset = HARDataset("test")
    train_subset, validation_subset = split_for_validation(train_dataset)

    train_dataloader = DataLoader(train_subset, batch_size=HYPERPARAMETERS["batch_size"], shuffle=True)
    validation_dataloader = DataLoader(validation_subset, batch_size=HYPERPARAMETERS["batch_size"], shuffle=False)
    #test_dataloader = Dataloader(train_subset, batch_size=HYPERPARAMETERS["batch_size"], shuffle=True)

    encoder = SignalEncoder()
    taskHead = TaskHead()
    model = nn.Sequential(encoder, taskHead)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=HYPERPARAMETERS["learning_rate"])


    print("Starting point")
    print("Train accuracy: {0}%".format(accuracy(model, train_dataloader)))
    print("Validation accuracy: {0}%\n".format(accuracy(model, validation_dataloader)))

    y_train_loss = []
    y_val_loss = []
    y_train_acc = []
    y_val_acc = []

    best_val_loss = float("inf")
    best_epoch = 0
    final_val_acc = 0
    best_model = None

    for epoch in tqdm(range(0, NUM_EPOCHS), desc="Training on " + str(NUM_EPOCHS) + " epochs"):
        # The model is set in training mode
        model.train()
    
        train_loss = 0
        # Iterating through the batches
        for _, data in enumerate(train_dataloader):
            inputs, labels = data
            
            # The optimization gradients are reset
            optimizer.zero_grad()
    
            # The current batch is forwarded to the network
            outputs = model(inputs)
            
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
            best_epoch = epoch + 1
            best_model = model.state_dict()


    x = np.linspace(1, NUM_EPOCHS, NUM_EPOCHS)
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

    print("Best validation loss:", best_val_loss)
    print("Final validation accuracy: ", str(final_val_acc) + "%")

    # Plotting the confusion matrices for train/validation splits
    plot_conf_matrix(model, train_dataloader, title="Confusion matrix related to the train split")
    plot_conf_matrix(model, validation_dataloader, title="Confusion matrix related to the validation split")

    return best_model




def concept_train ():
    pass


def save_model(best_model):
    if best_model is not None:
        torch.save(best_model, "model/weights.pt")
    else:
        print("No best model!")


if __name__ == "__main__":
    best_model = baseline_train()
    save_model(best_model)