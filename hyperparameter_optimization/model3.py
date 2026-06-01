import os
import argparse
from utils import float_to_unique_string
from grn_inference_msc.vae_model.data import Dataset
import torch
from torch.utils.tensorboard import SummaryWriter
from benchmark import stats_pipeline
from grn_inference_msc.zscore_variant.inference import get_A_1
import optuna
from grn_inference_msc.vae_model.vae_model import ModelConfig, Model
from grn_inference_msc.vae_model.constants import ProbabilityDistributions
from grn_inference_msc.vae_model.loss import BetaVAELoss
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--n_genes', required=True, type=int)
parser.add_argument('--n_reps', required=True, type=int)
parser.add_argument('--snr', required=True, type=float)

args = parser.parse_args()
#######################################################################
# Documentation
# The data used for the hyperparameter optimization should be in ./data
#######################################################################

# The datasets used for the hyperparameter search are in this directory
parent_dir = os.path.join(os.path.dirname(__file__), 'data')
# folder name of the dataset with the appropriate settings
dataset_folder_name = f'dataset_{args.n_genes}g_{args.n_reps}r_SNR{float_to_unique_string(args.snr)}'

dataset_full_path = os.path.join(parent_dir, dataset_folder_name)

if not os.path.exists(dataset_full_path):
    raise FileNotFoundError()

dataset = Dataset(dataset_full_path)

# Naming the study so it can easily be seen which dataset is being used
study_name = f'study_hyp_opt_model3_{args.n_genes}g_{args.n_reps}r_SNR{float_to_unique_string(args.snr)}'
storage_name = f'sqlite:///{os.path.dirname(__file__)}/RDB/{study_name}.db'


# the objective function to optimize
def objective(trial: optuna.Trial):
    # A writer to visualize in a tensorboard
    writer = SummaryWriter(
        os.path.join(
            os.path.dirname(__file__),
            f'hyp_opt_runs/{study_name}/trial-{trial.number}'
        )
    )

    # Suggest hyperparameters
    grn_layer_type = 'none'
    dim_latent = trial.suggest_categorical('dim_latent', [1, 2, 4, 8])

    # Network depth
    n_encoder_layers = 1
    n_decoder_layers = 1

    hidden_dim = trial.suggest_categorical('hidden_dim', [16, 32, 64, 128, 256])

    encoder_hidden_dimensions = [hidden_dim] * n_encoder_layers
    decoder_hidden_dimensions = [hidden_dim] * n_decoder_layers

    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)

    beta = trial.suggest_float('beta', 0.0001, 10, log=True)

    config = ModelConfig(
        n_genes=args.n_genes,
        dim_in=args.n_genes*args.n_reps,
        dim_latent=dim_latent,
        grn_layer_type=grn_layer_type,
        likelihood_distribution=ProbabilityDistributions.GAUSSIAN,
        encoder_hidden_dimensions=encoder_hidden_dimensions,
        decoder_hidden_dimensions=decoder_hidden_dimensions,
        device=device
    )
    model = Model(config)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = BetaVAELoss(beta)

    losses = []

    # Train the model
    for e in range(N_EPOCHS):
        optimizer.zero_grad()

        X, P = dataset.get_batch()
        X = torch.log(1 + X)# use log1p to stabilize variance

        try:
            mu, mu_latent, logvar_latent = model(X.to(device))
        except:
            raise RuntimeError()

        loss = criterion(X.to(device), mu, mu_latent, logvar_latent)
        loss.backward()
        losses.append(loss.item())

        optimizer.step()

    # Calculate the A matrix
    X, P = dataset.get_batch()
    X = torch.log(1 + X) # Use log1p to stabilize variance
    A = get_A_1(model, X, P, device)
    # Calculate the stats
    _, AUROC, _ = stats_pipeline(A, np.abs(np.sign(dataset.get_A().cpu().numpy())))

    return AUROC

N_EPOCHS = 1000 # Number of epochs to train the model for during the search
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Starting the hyperparameter search
study = optuna.create_study(
    study_name=study_name,
    storage=storage_name,
    direction='maximize',
    load_if_exists=True
)
study.optimize(objective, n_trials=100, n_jobs=1, catch=[RuntimeError])