## Þetta er skriptan sem loopar í gegnum allar stillingarnar og keyrir python scripturnar með réttum stillingum
import yaml
import os
import subprocess
from utils import float_to_unique_string
import torch
from grn_inference_msc.vae_model.data import Dataset
from grn_inference_msc.vae_model.vae_model import ModelConfig, Model
from grn_inference_msc.vae_model.constants import ProbabilityDistributions
from grn_inference_msc.vae_model.loss import BetaVAELoss
from grn_inference_msc.zscore_variant.inference import get_A_1
from tqdm import trange
from benchmark import stats_pipeline
import numpy as np
from grn_inference_msc.vae_model.early_stopping import EarlyStopping

with open(os.path.join(os.path.dirname(__file__), '../data_configurations.yml')) as f:
    data_settings = yaml.safe_load(f)

# This is done only for the following data configuration
n_genes = 100
n_reps = 3
snr = 0.1

# Use the following hyperparameters
beta = 0.00156743696817154
grn_layer_type = 'none'
encoder_hidden_dimensions = [256]
decoder_hidden_dimensions = [256]
lr = 0.000329347436257795
dim_latent = 4
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dataset_name = f'dataset_{n_genes}g_{n_reps}r_SNR{float_to_unique_string(snr)}'

# Use the hyperparameter optimization dataset as a validation dataset
validation_dataset_path = os.path.join(os.path.dirname(__file__), '../hyperparameter_optimization/data/', dataset_name)
dataset_val = Dataset(validation_dataset_path)

AUROCs = []
AUPRs = []
for i in range(50):
    dataset_name_i = f'{dataset_name}_{i}'
    dataset_path = os.path.join(os.path.dirname(__file__), '../compare_methods/data', dataset_name_i)
    dataset_i = Dataset(dataset_path)
    A_true = np.abs(np.sign(dataset_i.get_A().numpy()))

    # Create the model
    config = ModelConfig(
        n_genes=n_genes,
        dim_in=n_genes*n_reps,
        dim_latent=dim_latent,
        likelihood_distribution=ProbabilityDistributions.GAUSSIAN,
        encoder_hidden_dimensions=encoder_hidden_dimensions,
        decoder_hidden_dimensions=decoder_hidden_dimensions,
        device=device,
        grn_layer_type=grn_layer_type
    )

    model = Model(config)

    criterion = BetaVAELoss(beta)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    stopper = EarlyStopping(10, 2, model)

    # Train the model
    EPOCHS = trange(100)
    for e in EPOCHS:
        model.train()
        optimizer.zero_grad()

        X, P = dataset_i.get_batch_normalized()
        mu, mu_latent, logvar_latent = model(X.to(device))
        loss = criterion(X.to(device), mu, mu_latent, logvar_latent)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            X, P = dataset_val.get_batch_normalized()
            mu, mu_latent, logvar_latent = model(X.to(device))
            val_loss = criterion(X.to(device), mu, mu_latent, logvar_latent)

        EPOCHS.set_postfix(loss = loss.item(), val_loss=val_loss.item())

        if stopper.step(val_loss.item(), e):
            print (f'Early stopping at epoch {e}, best epoch was {stopper.best_epoch}')
            break

    # The model has been trained now collect the metrics
    model.load_state_dict(stopper.best_state)
    X, P = dataset_i.get_batch_normalized()
    A_est = get_A_1(model, X, P)
    AUPR, AUROC, _ = stats_pipeline(A_est, A_true)
    AUROCs.append(AUROC)
    AUPRs.append(AUPR)

print (np.mean(AUROCs))
print (2.01 * np.std(AUROCs, ddof=1) / np.sqrt(50))

print (np.mean(AUPRs))
print (2.01 * np.std(AUPRs, ddof=1) / np.sqrt(50))


