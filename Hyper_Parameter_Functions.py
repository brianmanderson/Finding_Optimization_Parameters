__author__ = 'Brian M Anderson'
# Created on 5/14/2020
from _collections import OrderedDict
import pandas as pd
import os
import numpy as np
from tensorboard.plugins.hparams import api as hp


def determine_if_in_excel(excel_path, run_data, features_list=None):
    if features_list is None:
        features_list = list(run_data.keys())
    data_frame = return_pandas_df(excel_path, features_list=features_list)
    trial_id = 0
    while trial_id in data_frame['Trial_ID'].values:
        trial_id += 1
    run_data['Trial_ID'] = trial_id
    current_run_df, features_list = return_current_df(run_data, features_list=data_frame.columns)
    if compare_base_current(data_frame=data_frame, current_run_df=current_run_df,
                            features_list=[i for i in data_frame.columns if i != 'Trial_ID']):
        print('Already done')
        return True
    else:
        print(current_run_df)
        data_frame = data_frame.append(current_run_df, ignore_index=True)
        data_frame.to_excel(excel_path, index=0)
        return False


def return_pandas_df(excel_path, features_list=['layers','filters','max_filters','min_lr','max_lr']):
    if not os.path.exists(excel_path):
        out_dict = OrderedDict()
        out_dict['Trial_ID'] = []
        for key in features_list:
            out_dict[key] = []
        df = pd.DataFrame(out_dict)
        df.to_excel(excel_path, index=0)
    else:
        df = pd.read_excel(excel_path)
    return df


def compare_base_current(data_frame, current_run_df, features_list):
    current_array = current_run_df[features_list].values
    base_array = data_frame[features_list].values
    if np.any(base_array) and np.max([np.min(i == current_array) for i in base_array]):
        return True
    return False


def return_current_df(run_data, features_list=['layers', 'filters', 'max_filters', 'min_lr', 'max_lr']):
    out_dict = OrderedDict()
    for feature in features_list:
        val = ''
        if feature in run_data:
            val = run_data[feature]
            if type(val) is bool:
                val = int(val)
            elif type(val) is tuple:
                val = val[0]
        out_dict[feature] = [val]
    out_features = [i for i in out_dict.keys()]
    return pd.DataFrame(out_dict), out_features


def return_hparams(run_data, features_list, excluded_keys=['iteration','save']):
    hparams = None
    for layer_key in features_list:
        break_out = False
        for exclude in excluded_keys:
            if layer_key.lower().find(exclude) != -1:
                break_out = True
        if break_out:
            continue
        if layer_key in run_data.keys():
            if hparams is None:
                hparams = OrderedDict()
            value = run_data[layer_key]
            if type(value) is tuple:
                value = value[0]
            hparams[hp.HParam(layer_key, hp.Discrete([run_data[layer_key]]))] = value
    return hparams