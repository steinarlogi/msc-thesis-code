import os
import argparse
from utils import float_to_unique_string
from grn_inference_msc.vae_model.data import Dataset
import torch
from torch.utils.tensorboard import SummaryWriter
from benchmark import stats_pipeline
import optuna
from grn_inference_msc.vae_model.vae_model import Model, ModelConfig
from grn_inference_msc.vae_model.constants import ProbabilityDistributions
from grn_inference_msc.vae_model.loss import VAEGaussianLikelihoodLoss
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
study_name = f'study_hyp_opt_{args.n_genes}g_{args.n_reps}r_SNR{float_to_unique_string(args.snr)}'
storage_name = f'sqlite:///{os.path.dirname(__file__)}/RDB/{study_name}.db'


# A function to add a list of scalars to the tensorboard
def add_scalars(writer: SummaryWriter, tag, values, x_values):
    for val, x in zip(values, x_values):
        writer.add_scalar(tag, val, x)


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
    grn_layer_type = 'before'
    dim_latent = trial.suggest_categorical('dim_latent', [1, 2, 4, 8])

    # Network depth
    n_encoder_layers = 1
    n_decoder_layers = 1

    hidden_dim = trial.suggest_categorical('hidden_dim', [16, 32, 64, 128, 256])

    encoder_hidden_dimensions = [hidden_dim] * n_encoder_layers
    decoder_hidden_dimensions = [hidden_dim] * n_decoder_layers

    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)

    beta = trial.suggest_float('beta', 0.0001, 10, log=True)
    alpha = trial.suggest_float('alpha', 0.0001, 10, log=True)

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
    criterion = VAEGaussianLikelihoodLoss(alpha, beta)

    losses = []
    AUROCs = []

    # Train the model and collect the AUROCs
    for e in range(N_EPOCHS):
        optimizer.zero_grad()

        X, P = dataset.get_batch()

        try:
            mu, mu_latent, logvar_latent = model(X.to(device))
        except:
            raise RuntimeError()

        loss = criterion(X.to(device), mu, mu_latent, logvar_latent, model.get_A())
        loss.backward()
        losses.append(loss.item())

        optimizer.step()

        AUPR, AUROC, stats = stats_pipeline(model.get_A().detach().cpu().numpy(), np.abs(np.sign(dataset.get_A().cpu().numpy())))

        # Write to the tensorboard
        writer.add_scalar('Metric/AUROC', AUROC, e)
        writer.add_scalar('Losses/loss', loss.item(), e)

        AUROCs.append(AUROC)

    smoothed_AUROCs = np.convolve(AUROCs, np.ones(51)/51, mode='valid')
    best_epoch = int(np.argmax(smoothed_AUROCs)) + 25 # Choose the best epoch as the one where the smoothed AUROC was maximized

    trial.set_user_attr('best_epoch', best_epoch)
    add_scalars(writer, 'Metric/AUROC_smoothed', smoothed_AUROCs, range(25, len(AUROCs) - 25))

    return max(smoothed_AUROCs)

N_EPOCHS = 500 # Number of epochs to train the model for during the search
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Starting the hyperparameter search
study = optuna.create_study(
    study_name=study_name,
    storage=storage_name,
    direction='maximize',
    load_if_exists=True
)
study.optimize(objective, n_trials=100, n_jobs=1, catch=[RuntimeError])