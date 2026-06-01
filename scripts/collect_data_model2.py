## Þetta er skriptan sem loopar í gegnum allar stillingarnar og keyrir python scripturnar með réttum stillingum
import yaml
import os
import subprocess


with open(os.path.join(os.path.dirname(__file__), '../data_configurations.yml')) as f:
    data_settings = yaml.safe_load(f)

for setting in data_settings['dataParams']:
    n_genes = setting['n_genes']
    n_reps = setting['n_reps']
    snr = setting['snr']

    hyp_opt_script_path = os.path.join(os.path.dirname(__file__), '../hyperparameter_optimization/model2.py')

    print (f'Running hyper parameter optimization for setting: n_genes: {n_genes}, n_reps: {n_reps}, snr: {snr}')

    try:
        subprocess.run([
            "python", hyp_opt_script_path,
            "--n_genes", str(n_genes),
            '--n_reps', str(n_reps),
            '--snr', str(snr)
        ],
        capture_output=True,
        text=True
        )
    except Exception as e:
        print (e)
        print ('failed to run script hyperparameter search')

    compare_methods_script_path = os.path.join(os.path.dirname(__file__), '../compare_methods/model2.py')

    print (f'Running compare methods for setting: n_genes: {n_genes}, n_reps: {n_reps}, snr: {snr}')

    try:
        subprocess.run([
            "python", compare_methods_script_path,
            "--n_genes", str(n_genes),
            '--n_reps', str(n_reps),
            '--snr', str(snr)
        ],
        capture_output=True,
        text=True
        )
    except:
        print ('failed to run script compare methods')
