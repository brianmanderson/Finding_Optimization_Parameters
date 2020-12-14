__author__ = 'Brian M Anderson'
# Created on 1/15/2020
import os
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.python.summary.summary_iterator import summary_iterator
import pandas as pd


def add_to_dictionary(path, all_dictionaries, path_id, fraction_start=0,
                      metric_name_and_criteria={'val_loss': np.min,'val_dice_coef_3D': np.max}, final_val=False):

    file_list = [i for i in os.listdir(path) if i.find('event') == 0]
    for file in file_list:
        k = summary_iterator(os.path.join(path, file))
        temp_dictionary = {}
        for event in k:
            for value in event.summary.value:
                if value.tag not in temp_dictionary:
                    temp_dictionary[value.tag] = []
                temp_dictionary[value.tag].append(value.simple_value)
        for metric_name in temp_dictionary.keys():
            if fraction_start == -1:
                metric = temp_dictionary[metric_name][-1]
            else:
                metric_values = temp_dictionary[metric_name]
                metric_values = metric_values[int(len(metric_values) * fraction_start):]
                if metric_name in metric_name_and_criteria.keys():
                    metric = metric_name_and_criteria[metric_name](metric_values)
                else:
                    if metric_name.find('accuracy') != -1 or metric_name.find('dice') != -1 or \
                            metric_name.find('dsc') != -1:
                        metric = np.max(metric_values)
                    else:
                        metric = np.min(metric_values)
            temp_dictionary[metric_name] = metric
        all_dictionaries[path_id] = temp_dictionary
        del temp_dictionary
    return all_dictionaries


def iterate_paths_add_to_dictionary(path, all_dictionaries, final_val=False, fraction_start=0,
                                    metric_name_and_criteria={'epoch_loss': np.min,'val_dice_coef_3D': np.max}):
    """
    :param path: path to .event files from tensorflow training
    :param all_dictionaries: a dictionary which will have all of the run dictionaries
    :param final_val: if you want to take the 'final value' of training. Being removed for fraction_start
    :param fraction_start: fraction along the training processes at which you want to start metric analysis. Use 0 to
    say you want the entire run. Use -1 to say only the last value
    :param metric_name_and_criteria: a dictionary of {'metric': 'loss criteria'} to evaluate results
    :return: dictionary full of metric dictionaries from runs
    """
    for root, folders, files in os.walk(path):
        event_files = [i for i in files if i.find('event') == 0]
        if event_files and root.find('validation') != -1:
            path_id = os.path.split(os.path.split(root)[0])[-1]
            try:
                print(root)
                add_to_dictionary(root, all_dictionaries, path_id=path_id, fraction_start=fraction_start,
                                  metric_name_and_criteria=metric_name_and_criteria, final_val=final_val)
            except:
                return None
    return all_dictionaries


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
    iterate_paths_add_to_dictionary(input_path, all_dictionaries, metric_name_and_criteria=metric_name_and_criteria,
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
