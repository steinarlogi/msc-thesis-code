import os
import argparse
from utils import float_to_unique_string
import sqlite3
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
from grn_inference_msc.vae_model.data import Dataset
from benchmark import stats_pipeline
from basic_algorithms import zscore, least_squares
from grn_inference_msc.vae_model.vae_model import ConditionalModelConfig, ConditionalModel
from grn_inference_msc.vae_model.constants import ProbabilityDistributions
import torch
from grn_inference_msc.vae_model.loss import VAEGaussianLikelihoodLoss
import matplotlib.pyplot as plt
import tqdm

## Þetta er skriftan sem tekur bestu parametrana sem fundust fyrir ákveðið dataset
# og hversu lengi módelið var þjálfað á því og þjálfar módelið á 50 svoleiðis datasettum og plottar
# Aurocið og AUPR-ið og sem bar chart og ber saman við least squares? og zscore

# Hvað þarf að gera í skriftunni

# stillingarnar á gagnasettunum sem er verið að vinna með þurfa að koma inn sem argument.
parser = argparse.ArgumentParser()

parser.add_argument('--n_genes', required=True, type=int)
parser.add_argument('--n_reps', required=True, type=int)
parser.add_argument('--snr', required=True, type=float)

args = parser.parse_args()

# Næst þarf ég að fá inn bestu parametrana sem eru vistaðir í gagnagrunninn undir hyp_opt_studiesRDB
# Nafnið á rétta grunninum er fundið með stillingunum á gögnunum
study_name = f'study_hyp_opt_model2_{args.n_genes}g_{args.n_reps}r_SNR{float_to_unique_string(args.snr)}'
database_path = os.path.join(os.path.dirname(__file__), '../hyperparameter_optimization/RDB', f'{study_name}.db')

database_conn = sqlite3.connect(database_path)
cur = database_conn.cursor()

res = cur.execute('select param_name, param_value from trial_params where trial_id = (select trial_id from trial_values where value >= (select max(value) from trial_values) limit 1)')
param_dict = {r[0]: r[1] for r in res.fetchall()}
# Get the epoch vale:
res = cur.execute('SELECT value_json FROM trial_user_attributes WHERE trial_id=(select trial_id from trial_values where value >= (select max(value) from trial_values) limit 1)')
epoch = res.fetchone()[0]

# Mappings for the values that are categorical in the hyp opt
dim_latent_cat_mapping = [1, 2, 4, 8]
hidden_dim_cat_mapping = [16, 32, 64, 128, 256]

# Extract the best parameters
alpha = float(param_dict['alpha'])
beta = float(param_dict['beta'])
dim_latent = dim_latent_cat_mapping[int(param_dict['dim_latent'])]
grn_layer_type = 'before'
hidden_dim = hidden_dim_cat_mapping[int(param_dict['hidden_dim'])]
lr = float(param_dict['lr'])
n_decoder_layers = 1
n_encoder_layers = 1
encoder_hidden_dimensions = [hidden_dim] * n_encoder_layers
decoder_hidden_dimensions = [hidden_dim] * n_decoder_layers
n_epochs = int(epoch)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

trial_number = cur.execute('select trials.number from trials join trial_values on trial_values.trial_id = trials.trial_id where value = (select max(value) from trial_values) limit 1').fetchall()
trial_number = int(trial_number[0][0])

# Close the database connection
database_conn.close()

# Loop through all the datasets that exist with the specified parameters and get the results on that
autoencoder_AUROCs = []
ls_AUROCs = []
zscore_AUROCs = []

autoencoder_AUPRs = []
ls_AUPRs = []
zscore_AUPRs = []

for i in tqdm.tqdm(range(50), desc='Comparing methods'):
    dataset_name = f'dataset_{args.n_genes}g_{args.n_reps}r_SNR{float_to_unique_string(args.snr)}_{i}'
    dataset_full_path = os.path.join(
        os.path.dirname(__file__),
        'data',
        dataset_name
    )

    # Load the dataset
    dataset = Dataset(dataset_full_path)
    true_adjacency = np.abs(np.sign(dataset.get_A().numpy()))

    X, P = dataset.get_batch()
    X = torch.log(1 + X) # Use log1p to stabilize variance

    ls_solution = least_squares(X.numpy().T, P.numpy().T)
    zscore_solution = zscore(X.numpy().T, P.numpy().T)

    ls_AUPR, ls_AUROC, ls_stats = stats_pipeline(ls_solution, true_adjacency)
    zscore_AUPR, zscore_AUROC, zscore_stats = stats_pipeline(zscore_solution, true_adjacency)

    ls_AUROCs.append(ls_AUROC)
    zscore_AUROCs.append(zscore_AUROC)

    ls_AUPRs.append(ls_AUPR)
    zscore_AUPRs.append(zscore_AUPR)

    # Now the training loop for the model
    model_config = ConditionalModelConfig(
        dim_condition=args.n_genes*args.n_reps,
        n_genes=args.n_genes,
        dim_in=args.n_genes * args.n_reps,
        grn_layer_type=grn_layer_type,
        dim_latent=dim_latent,
        likelihood_distribution=ProbabilityDistributions.GAUSSIAN,
        encoder_hidden_dimensions=encoder_hidden_dimensions,
        decoder_hidden_dimensions=decoder_hidden_dimensions,
        device=device
    )
    model = ConditionalModel(model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = VAEGaussianLikelihoodLoss(alpha, beta)

    for e in range(n_epochs):
        optimizer.zero_grad()

        mu, mu_latent, logvar_latent = model(X.to(device), P.to(device))
        loss = criterion(X.to(device), mu, mu_latent, logvar_latent, model.get_A())
        loss.backward()

        optimizer.step()

    autoencoder_solution = model.get_A()
    autoencoder_AUPR, autoencoder_AUROC, autoencoder_stats = stats_pipeline(autoencoder_solution.detach().cpu().numpy(), true_adjacency)

    autoencoder_AUPRs.append(autoencoder_AUPR)
    autoencoder_AUROCs.append(autoencoder_AUROC)


## Finally save the data found

# Plot boxplot to compare AUROCs
plt.title('AUROC comparison between methods')
plt.boxplot([ls_AUROCs, zscore_AUROCs, autoencoder_AUROCs], labels=['Least squares', 'zscore', 'Autoencoder'])
plt.hlines(0.5, xmin=plt.gca().get_xlim()[0], xmax=plt.gca().get_xlim()[1], linestyles='dotted', colors='blue')
plt.savefig(os.path.join(os.path.dirname(__file__), 'results_model2', 'plots', f'box_comp_methods_dataset_{args.n_genes}g_{args.n_reps}r_SNR{float_to_unique_string(args.snr)}.png'))

plt.cla()

# Plot boxplot to compare AUPRs
plt.title('AUPR comparison between methods')
plt.boxplot([ls_AUPRs, zscore_AUPRs, autoencoder_AUPRs], labels=['Least squares', 'zscore', 'Autoencoder'])
plt.savefig(os.path.join(os.path.dirname(__file__), 'results_model2', 'plots', f'box_comp_methods_AUPR_dataset_{args.n_genes}g_{args.n_reps}r_SNR{float_to_unique_string(args.snr)}.png'))

plt.cla()

np.save(os.path.join(os.path.dirname(__file__), 'results_model2', 'raw_data', f'autoencoder_AUROCs_{args.n_genes}g_{args.n_reps}r_SNR{float_to_unique_string(args.snr)}'), autoencoder_AUROCs)
np.save(os.path.join(os.path.dirname(__file__), 'results_model2', 'raw_data', f'ls_AUROCs_{args.n_genes}g_{args.n_reps}r_SNR{float_to_unique_string(args.snr)}'), ls_AUROCs)
np.save(os.path.join(os.path.dirname(__file__), 'results_model2', 'raw_data', f'zscore_AUROCs_{args.n_genes}g_{args.n_reps}r_SNR{float_to_unique_string(args.snr)}'), zscore_AUROCs)

np.save(os.path.join(os.path.dirname(__file__), 'results_model2', 'raw_data', f'autoencoder_AUPRs_{args.n_genes}g_{args.n_reps}r_SNR{float_to_unique_string(args.snr)}'), autoencoder_AUPRs)
np.save(os.path.join(os.path.dirname(__file__), 'results_model2', 'raw_data', f'ls_AUPRs_{args.n_genes}g_{args.n_reps}r_SNR{float_to_unique_string(args.snr)}'), ls_AUPRs)
np.save(os.path.join(os.path.dirname(__file__), 'results_model2', 'raw_data', f'zscore_AUPRs_{args.n_genes}g_{args.n_reps}r_SNR{float_to_unique_string(args.snr)}'), zscore_AUPRs)
