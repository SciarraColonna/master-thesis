import torch
import numpy as np
from matplotlib import pyplot as plt
import random


PATHS = {
    "train": "data/UCI-HAR/train/",
    "test": "data/UCI-HAR/test/"
}

NUM_CONCEPTS = 3
WINDOW_TIMESTEPS = 128
K = 3


"""
Function that performs a K-means clustering algorithm on the set of variances associated with the X component of the 
body acceleration. The clustering is performed only considering the training samples and not the test samples. 
The function returns the centroids produced by the clustering algorithm.
"""
def get_centroids ():
    tot_acc_x = np.loadtxt("{0}Inertial Signals/body_acc_x_{1}.txt".format(PATHS["train"], "train"))
    activities = np.loadtxt("{0}y_{1}.txt".format(PATHS["train"], "train"))

    variances = list()

    for idx in range(0, len(activities)):
        var = np.std((tot_acc_x[idx])).item()
        variances.append(var)

    # K-means algorithm
    num_iterations = 50

    # List of CV values to cluster (one for each dynamic activity)
    variances_tensor = (torch.tensor(variances)).view((len(variances), 1))
    # Initialization of the centroids
    centroids = torch.tensor(random.sample(variances, K)).view((K, 1))

    for j in range(0, num_iterations):
        # Euclidean distance between each variance and each centroid
        distances = torch.cdist(variances_tensor, centroids)
        _, cluster_labels = torch.min(distances, dim=1)

        # A new centroid is calculated for each cluster
        for idx in range(0, K):
            if torch.sum(cluster_labels == idx) > 0:
                centroids[idx] = torch.mean(variances_tensor[cluster_labels == idx], dim=0)

    # The centroids are sorted in order to facilitate the cluster asignment
    centroids, _ = torch.sort(centroids, dim=0)

    return centroids


"""
Function that clears the previous content of the concept files (if any)
"""
def clear_concepts (type):
    # Clearing the previous content of the concept files (if any)
    concepts_file = open("{0}concepts_{1}.txt".format(PATHS[type], type), "w")
    concepts_file.write("")
    concepts_file.close()


"""
Function that performs the concept-labeling for each sample of both the training and the test set.
"""
def concept_labeling (type, centroids):
    concepts_file = open("{0}concepts_{1}.txt".format(PATHS[type], type), "a")

    activities = np.loadtxt("{0}y_{1}.txt".format(PATHS[type], type))
    tot_acc_x = np.loadtxt("{0}Inertial Signals/total_acc_x_{1}.txt".format(PATHS[type], type))
    tot_acc_y = np.loadtxt("{0}Inertial Signals/total_acc_y_{1}.txt".format(PATHS[type], type))
    tot_acc_z = np.loadtxt("{0}Inertial Signals/total_acc_z_{1}.txt".format(PATHS[type], type))

    concepts = np.empty((len(activities), NUM_CONCEPTS))

    for idx in range(0, len(concepts)):
        # Labeling dynamic (1) or static (0) for classes [1, 3] and [4, 6]
        if int(activities[idx]) <= 3:
            concepts[idx][0] = 1
        else:
            concepts[idx][0] = 0

        # Labeling horizontal posture (0, lying) or vertical posture (1, all the others) 
        x_sum = 0
        y_sum = 0
        z_sum = 0
        for timestep in range(0, WINDOW_TIMESTEPS):
            x_sum += tot_acc_x[idx][timestep]
            y_sum += tot_acc_y[idx][timestep]
            z_sum += tot_acc_z[idx][timestep]

        if abs(y_sum) < abs(x_sum) and abs(z_sum) < abs(x_sum):
            concepts[idx][1] = 1
        else:
            concepts[idx][1] = 0

        # Labeling the "energy level" of the X component of the body acceleration, categorizing it with respect to
        # three possible values (0, 1 and 2 for low, medium and high)
        if activities[idx] > 3:
            concepts[idx][2] = 0
        else:
            variance = np.std(tot_acc_x[idx])

            distances = np.empty((K,))
            for i in range(0, K):
                distances[i] = abs(centroids[i] - variance)

            concepts[idx][2] = np.argmin(distances)


    # The calculated concepts are written on the related txt file
    for idx in range(0, len(concepts)):
        for concept in concepts[idx]:
            concepts_file.write(str(concept) + "  ")
        concepts_file.write("\n")

    concepts_file.close()

    
if __name__ == "__main__":
    centroids = get_centroids()

    clear_concepts("train")
    clear_concepts("test")

    concept_labeling("train", centroids)
    concept_labeling("test", centroids)