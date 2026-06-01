import numpy as np
import torch

def get_latent_representation(model, X):
    x_out_encoder = model.encoder(X)
    mu_latent = model.encoder_head_mu(x_out_encoder)
    return mu_latent


def get_A_1(model, X, P, device=torch.device('cuda')):
    """A method to infer A with the VAE based on the zscore method
    It first finds the latent represenation of the vector containing only the experiments
    where the regulator has been perturbed. Then it finds the latent representation of the
    vector containing the expression of the target gene in all the experiments
    and then calculates the norm of the differences

    Returns:
        (ndarray): The weighted adjacency matrix
    """
    n_genes = X.shape[0]

    A = np.zeros((n_genes, n_genes))
    Z = get_latent_representation(model, X.to(device))

    for m in range(n_genes):
        Z_m = get_latent_representation(model, (X * P[m].unsqueeze(0).expand((n_genes, -1))).to(device))
        A[m, :] = (torch.linalg.norm((Z - Z_m), dim=1) / torch.linalg.norm(Z, dim=1)).detach().cpu().numpy()

    return A