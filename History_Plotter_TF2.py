__author__ = 'Brian M Anderson'
# Created on 1/15/2020
import os
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.python.summary.summary_iterator import summary_iterator
import pandas as pd


def add_to_dictionary(path, all_dictionaries, path_id,
                      metric_name_and_criteria={'val_loss':np.min,'val_dice_coef_3D':np.max}, final_val=False):
    path = path.lower()
    file_list = [i for i in os.listdir(path) if i.find('event') == 0]
    for file in file_list:
        k = summary_iterator(os.path.join(path,file))
        temp_dictionary = {}
        for event in k:
            for value in event.summary.value:
                if value.tag not in temp_dictionary:
                    temp_dictionary[value.tag] = []
                temp_dictionary[value.tag].append(value.simple_value)
        for metric_name in temp_dictionary.keys():
            if metric_name in metric_name_and_criteria.keys():
                metric = metric_name_and_criteria[metric_name](temp_dictionary[metric_name])
            else:
                if final_val:
                    metric = temp_dictionary[metric_name][-1]
                elif metric_name.find('accuracy') != -1 or metric_name.find('dice') != -1 or metric_name.find('dsc') != -1:
                    metric = np.max(temp_dictionary[metric_name])
                else:
                    metric = np.min(temp_dictionary[metric_name])
            temp_dictionary[metric_name] = metric
        all_dictionaries[path_id] = temp_dictionary
        del temp_dictionary
    return all_dictionaries


def create_dictionary_from_path(path, all_dictionaries, final_val=False,
                                metric_name_and_criteria={'epoch_loss':np.min,'val_dice_coef_3D':np.max}):
    files = []
    folders = []
    for root, folders, files in os.walk(path):
        break
    event_files = [i for i in files if i.find('event') == 0]
    if event_files and path.find('validation') != -1:
        path_id = path.split('\\')
        if len(path_id) == 1:
            path_id = path.split('/')
        path_id = path_id[-2]
        try:
            print(path)
            add_to_dictionary(path, all_dictionaries,path_id=path_id,
                              metric_name_and_criteria=metric_name_and_criteria, final_val=final_val)
        except:
            return None
    for folder in folders:
        create_dictionary_from_path(os.path.join(path, folder), all_dictionaries,
                                    metric_name_and_criteria=metric_name_and_criteria, final_val=final_val)
    return None


def complete_dictionary(all_dictionaries):
    # Get the names of all variables, we will need to make a complete list
    out_dictionary = {'Trial_ID':[]}
    for trial_id in all_dictionaries:
        try:
            out_dictionary['Trial_ID'].append(int(trial_id.split('_')[-1]))
        except:
            print('{} is not valid'.format(trial_id))
            continue
        for key in all_dictionaries[trial_id]:
            if key not in out_dictionary:
                out_dictionary[key] = []
            out_dictionary[key].append(all_dictionaries[trial_id][key])
    return out_dictionary


def combine_hparamxlsx_and_metricxlsx(hparameter_excel_sheet, total_dictionary_xlsx_path, out_path):
    metric_data = pd.read_excel(total_dictionary_xlsx_path)
    metric_data = metric_data.dropna()

    hparameter_data = pd.read_excel(hparameter_excel_sheet)
    hparameter_data = hparameter_data.dropna()

    combined_df = pd.merge(hparameter_data, metric_data, on='Trial_ID')
    combined_df.to_excel(out_path, index=0)
    return None


def create_excel_from_event(input_path=None, excel_out_path=os.path.join('.','Model_Optimization.xlsx'),
                            metric_name_and_criteria={'val_loss':np.min,'val_dice_coef_3D':np.max},
                            final_val=False):
    '''
    :param input_path: folder header path to Trial_IDs
    :param excel_out_path: path to write excel sheet
    :param metric_name_and_criteria: optional, dictionary of value:metric
    :param final_val: only take the final value of training?
    :return:
    '''
    if input_path is None:
        return None
    all_dictionaries = {}
    create_dictionary_from_path(input_path, all_dictionaries, metric_name_and_criteria=metric_name_and_criteria,
                                final_val=final_val)
    total_dictionary = complete_dictionary(all_dictionaries)
    df = pd.DataFrame(total_dictionary)
    df.to_excel(excel_out_path, index=0)
    return None


def plot_from_excel(excel_path, variable_name='layers', metric_name='val_loss', log_val=True, criterias=[]):
    # criteria_base = lambda x, variable_name, value: np.asarray(list(x[variable_name].values())) == value
    # criteria = [partial(criteria_base, variable_name='layers', value=5)]
    variable_name = variable_name.lower()
    data = pd.read_excel(excel_path).to_dict()
    y = np.asarray(list(data[metric_name].values()))
    x = np.asarray(list(data[variable_name].values()))
    keep = np.ones(x.shape)
    for criteria in criterias:
        values = criteria(data)
        keep[~values] = 0
    y = y[keep == 1]
    x = x[keep == 1]
    if log_val:
        plt.yscale('log')
    plt.scatter(x, y)
    plt.xlabel(variable_name)
    plt.ylabel(metric_name)
    plt.title('{} vs {}'.format(metric_name,variable_name))
    plt.show()
    return None


def main():
    pass


if __name__ == '__main__':
    main()
