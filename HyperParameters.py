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
        df = pd.read_excel(excel_path, engine='openpyxl')
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
        value = run_data[layer_key]
        if type(value) is np.int64 or type(value) is np.bool_:
            value = int(value)
        elif type(value) is np.float64:
            value = float(value)
        if layer_key in run_data.keys():
            if hparams is None:
                hparams = OrderedDict()
            hparams[hp.HParam(layer_key, hp.Discrete([value]))] = value
    return hparams


def is_df_within_another(data_frame, current_run_df, features_list=('Model_Type', 'min_lr', 'max_lr')):
    """
    :param data_frame: a base data_frame from an excel sheet
    :param current_run_df: current run dataframe
    :param features_list: list or tuple of features to compare along
    :return: Boolean for if it is contained
    """
    current_array = current_run_df[list(features_list)].values
    base_array = data_frame[list(features_list)].values
    if np.any(base_array) and np.max([np.min(i == current_array) for i in base_array]):
        return True
    return False


if __name__ == '__main__':
    pass
