import torch
import os

class Dataset:
    def __init__(self, path):
        """This is a class used as a container for the grn datasets"""
        self.P = torch.load(f'{os.path.join(path, "P.pt")}', weights_only=False).T
        self.Y = torch.load(f'{os.path.join(path, "Y.pt")}', weights_only=False).T
        self.A = torch.load(f'{os.path.join(path, "true_network.pt")}', weights_only=False)


    def get_batch(self):
        """A function that returns a tuple containing two items
        First the whole gene expression data, and second the whole perturbation
        design represented as a one-hot encoded vector where the gene
        being perturbed is 1 and the other genes have value 0
        """
        one_hot_encoded_P = torch.zeros_like(self.P)
        one_hot_encoded_P[self.P != 0] = 1

        return self.Y, one_hot_encoded_P


    def __getitem__(self, i):
        return self.Y[i], self.P[i]


    def get_A(self):
        return self.A
