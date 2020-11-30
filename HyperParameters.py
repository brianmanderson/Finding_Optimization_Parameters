__author__ = 'Brian M Anderson'
# Created on 11/28/2020
from _collections import OrderedDict
from tensorboard.plugins.hparams import api as hp
import os
import pandas as pd
import numpy as np


def return_pandas_df(excel_path, features_list=('layers', 'filters', 'max_filters', 'min_lr', 'max_lr')):
    """
    :param excel_path: path to an excel sheet
    :param features_list: list of features to make a dataframe of
    :return: pandas dataframe
    """
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


def return_hparams(run_data, features_list, excluded_keys=('iteration', 'save')):
    """
    :param run_data: a dictionary that defines the current iteration run
    :param features_list: list of features
    :param excluded_keys: keys to exclude when comparing parameters
    :return:
    """
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
            hparams[hp.HParam(layer_key, hp.Discrete([run_data[layer_key]]))] = run_data[layer_key]
    return hparams


def is_current_df_in_base_df(data_frame, current_run_df, features_list=('Model_Type', 'min_lr', 'max_lr')):
    current_array = current_run_df[features_list].values
    base_array = data_frame[features_list].values
    if np.any(base_array) and np.max([np.min(i == current_array) for i in base_array]):
        return True
    return False


if __name__ == '__main__':
    pass
