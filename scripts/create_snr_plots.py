import argparse
import os
import yaml
from utils import float_to_unique_string
import numpy as np
import matplotlib.pyplot as plt


def pretty_print_table(table_data):
    print (''.join(f'{x:<20}' for x in table_data[0]))
    print ('-'*100)
    for i in range(1,len(table_data)):
        print (''.join(f'{x:<20.4f}' for x in table_data[i]))


parser = argparse.ArgumentParser()
parser.add_argument('--raw_data_dir', required=True, type=str)
parser.add_argument('--data_configurations_dir', required=True, type=str)

args = parser.parse_args()

with open(args.data_configurations_dir) as f:
    data_configurations = yaml.safe_load(f)

print ('Note that the confidence intervals are only correct when the results contain 50 samples')
print ('-'*20)

x, y, y_zscore = [], [], []
for data_config in data_configurations['dataParams']:
    file_identifier = f'{data_config['n_genes']}g_{data_config['n_reps']}r_SNR{float_to_unique_string(data_config['snr'])}'
    autoencoder_AUROCs_path = os.path.join(args.raw_data_dir, f'autoencoder_AUROCs_{file_identifier}.npy')
    autoencoder_AUPRs_path = os.path.join(args.raw_data_dir, f'autoencoder_AUPRs_{file_identifier}.npy')
    ls_AUROCs_path = os.path.join(args.raw_data_dir, f'ls_AUROCs_{file_identifier}.npy')
    ls_AUPRs_path = os.path.join(args.raw_data_dir, f'ls_AUPRs_{file_identifier}.npy')
    zscore_AUROCs_path = os.path.join(args.raw_data_dir, f'zscore_AUROCs_{file_identifier}.npy')
    zscore_AUPRs_path = os.path.join(args.raw_data_dir, f'zscore_AUPRs_{file_identifier}.npy')

    autoencoder_AUROCs_mean = np.mean(np.load(os.path.join(autoencoder_AUROCs_path)))
    autoencoder_AUROCs_ci = 2.01 * np.std(np.load(os.path.join(autoencoder_AUROCs_path)), ddof=1) / np.sqrt(50)
    autoencoder_AUPRs_mean = np.mean(np.load(os.path.join(autoencoder_AUPRs_path)))
    autoencoder_AUPRs_ci = 2.01 * np.std(np.load(os.path.join(autoencoder_AUPRs_path)), ddof=1) / np.sqrt(50)
    ls_AUROCs_mean = np.mean(np.load(os.path.join(ls_AUROCs_path)))
    ls_AUROCs_ci = 2.01 * np.std(np.load(os.path.join(ls_AUROCs_path)), ddof=1) / np.sqrt(50)
    ls_AUPRs_mean = np.mean(np.load(os.path.join(ls_AUPRs_path)))
    ls_AUPRs_ci = 2.01 * np.std(np.load(os.path.join(ls_AUPRs_path)), ddof=1) / np.sqrt(50)
    zscore_AUROCs_mean = np.mean(np.load(os.path.join(zscore_AUROCs_path)))
    zscore_AUROCs_ci = 2.01 * np.std(np.load(os.path.join(zscore_AUROCs_path)), ddof=1) / np.sqrt(50)
    zscore_AUPRs_mean = np.mean(np.load(os.path.join(zscore_AUPRs_path)))
    zscore_AUPRs_ci = 2.01 * np.std(np.load(os.path.join(zscore_AUPRs_path)), ddof=1) / np.sqrt(50)

    if data_config['n_reps'] != 3:
        continue

    x.append(data_config['snr'])
    y.append(autoencoder_AUROCs_mean)
    y_zscore.append(zscore_AUROCs_mean)

x, y, y_zscore = np.array(x), np.array(y), np.array(y_zscore)
idxs = np.argsort(x)

plt.title('The effect of noise on AUROC')
plt.semilogx(x[idxs], y[idxs], color='blue', label='Model 3')
plt.semilogx(x[idxs], y_zscore[idxs], color='red', label='z score')
plt.legend()
plt.xlabel('SNR')
plt.ylabel('AUROC')
plt.show()