import argparse
import os
import yaml
from utils import float_to_unique_string
import numpy as np

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

table_headers = ['n_genes', 'n_reps', 'snr', 'ls', 'lsci', 'zscore', 'zscore ci', 'custom_method', 'custom_method ci']
table_data = [table_headers]

print ('Note that the confidence intervals are only correct when the results contain 50 samples')
print ('-'*20)

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


    line = [data_config['n_genes'], data_config['n_reps'], data_config['snr'], ls_AUPRs_mean, ls_AUPRs_ci, zscore_AUPRs_mean, zscore_AUPRs_ci, autoencoder_AUPRs_mean, autoencoder_AUPRs_ci]
    table_data.append(line)


pretty_print_table(table_data)