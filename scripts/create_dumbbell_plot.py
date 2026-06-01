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
parser.add_argument('--raw_data_dir_model1', required=True, type=str)
parser.add_argument('--raw_data_dir_model2', required=True, type=str)
parser.add_argument('--data_configuration', required=True, type=str)

args = parser.parse_args()

with open(args.data_configuration) as f:
    data_configurations = yaml.safe_load(f)

print ('Note that the confidence intervals are only correct when the results contain 50 samples')
print ('-'*20)

autoencoder1_data, autoencoder2_data = [], []
for data_config in data_configurations['dataParams']:
    file_identifier = f'{data_config['n_genes']}g_{data_config['n_reps']}r_SNR{float_to_unique_string(data_config['snr'])}'
    autoencoder1_AUROCs_path = os.path.join(args.raw_data_dir_model1, f'autoencoder_AUROCs_{file_identifier}.npy')
    autoencoder2_AUROCs_path = os.path.join(args.raw_data_dir_model2, f'autoencoder_AUROCs_{file_identifier}.npy')

    autoencoder1_AUROCs_mean = np.mean(np.load(os.path.join(autoencoder1_AUROCs_path)))
    autoencoder1_AUROCs_ci = 2.01 * np.std(np.load(os.path.join(autoencoder1_AUROCs_path)), ddof=1) / np.sqrt(50)
    autoencoder2_AUROCs_mean = np.mean(np.load(os.path.join(autoencoder2_AUROCs_path)))
    autoencoder2_AUROCs_ci = 2.01 * np.std(np.load(os.path.join(autoencoder2_AUROCs_path)), ddof=1) / np.sqrt(50)

    autoencoder1_data.append(autoencoder1_AUROCs_mean)
    autoencoder2_data.append(autoencoder2_AUROCs_mean)

y = list(range(1, len(autoencoder1_data) + 1))

print (autoencoder1_data)
print (autoencoder2_data)
print (list(map(lambda x: min(x), zip(autoencoder1_data))))

plt.title('Comparison of the AUROC of model 1 and model 2')
plt.scatter(autoencoder1_data, y, s=80, marker='o', color='royalblue', label='Model 1')
plt.scatter(autoencoder2_data, y, s=80, marker='o', color='indianred', label='Model 2')
plt.yticks(y)
plt.ylabel('Dataset configuration')
plt.xlabel('AUROC')
plt.legend()
plt.hlines(y, list(map(lambda x: min(x), zip(autoencoder1_data, autoencoder2_data))), list(map(lambda x: max(x), zip(autoencoder1_data, autoencoder2_data))), color='lightgray')
plt.grid(axis='x', linestyle='--', alpha=0.4)

plt.show()
