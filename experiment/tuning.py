import os
import time
import yaml
from pathlib import Path

from utils.io import load_yaml


def hyper_parameter_tuning(train, validation, test, params, dataset, table_save_path, input_dim=11, problem=None,
                           seed=2021, gpu_on=True, save_model=False, feature_idx=None):
    table_path = load_yaml('config/global.yml', key='path')['tables']
    model_path = load_yaml('config/global.yml', key='path')['models']

    if not os.path.exists(model_path + dataset):
        os.makedirs(model_path + dataset)

    algorithm, trials, best_params = None, None, None
    for algorithm in params['models']:
        trials, best_params = params['models'][algorithm](train,
                                                          validation,
                                                          test,
                                                          model_pl=model_path + dataset,
                                                          iteration=params['iter'],
                                                          seed=seed,
                                                          gpu_on=gpu_on,
                                                          save_model=save_model,
                                                          input_dim=input_dim,
                                                          feature_idx=feature_idx)
    if not os.path.exists(table_path + table_save_path):
        if not os.path.exists(table_path + dataset):
            os.makedirs(table_path + dataset)

    trials.to_csv(table_path + table_save_path)

    if Path(table_path + dataset + 'op_hyper_params.yml').exists():
        pass
    else:
        if dataset == 'synthetic_data_syn1/':
            yaml.dump(dict(synthetic_data_syn1=dict()),
                      open(table_path + dataset + 'op_hyper_params.yml', 'w'), default_flow_style=False)
        elif dataset == 'synthetic_data_syn3/':
            yaml.dump(dict(synthetic_data_syn3=dict()),
                      open(table_path + dataset + 'op_hyper_params.yml', 'w'), default_flow_style=False)
        elif dataset == 'synthetic_data_syn5/':
            yaml.dump(dict(synthetic_data_syn5=dict()),
                      open(table_path + dataset + 'op_hyper_params.yml', 'w'), default_flow_style=False)
        elif dataset == 'coat/':
            yaml.dump(dict(coat=dict()),
                      open(table_path + dataset + 'op_hyper_params.yml', 'w'), default_flow_style=False)
        elif dataset == 'adult/':
            yaml.dump(dict(adult=dict()),
                      open(table_path + dataset + 'op_hyper_params.yml', 'w'), default_flow_style=False)
        elif dataset == 'coat_uniform/':
            yaml.dump(dict(coat_uniform=dict()),
                      open(table_path + dataset + 'op_hyper_params.yml', 'w'), default_flow_style=False)

    time.sleep(0.5)
    hyper_params_dict = yaml.safe_load(open(table_path + dataset + 'op_hyper_params.yml', 'r'))
    hyper_params_dict[problem.replace('/', '')][algorithm] = best_params
    yaml.dump(hyper_params_dict, open(table_path + dataset + 'op_hyper_params.yml', 'w'),
              default_flow_style=False)
