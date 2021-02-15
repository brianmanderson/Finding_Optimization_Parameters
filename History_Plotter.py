__author__ = 'Brian M Anderson'
# Created on 1/15/2020
import sys, os, pickle
from .Plot_Best_Learning_rates import save_obj, load_obj, np, plt
from tensorflow.python.summary.summary_iterator import summary_iterator
import pandas as pd
from functools import partial


def add_to_dictionary(path,all_dictionaries,names=[],metric_name_and_criteria={'val_loss':np.min,'val_dice_coef_3D':np.max}):
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
        values = {}
        names = [i.lower() for i in names]
        for name in names:
            if name in path:
                val = path.split(name)[0].split('\\')[-1]
                if val[-1] == '_':
                    val = val[:-1]
                values[name] = val.split('_')[-1]
        full_name = ''.join([values[i]+i for i in values])
        values['Full_Name'] = full_name
        for metric_name in metric_name_and_criteria:
            if metric_name in temp_dictionary:
                metric = metric_name_and_criteria[metric_name](temp_dictionary[metric_name])
                values[metric_name] = metric
        all_dictionaries.append(values)
    return all_dictionaries


def down_folder(path, all_dictionaries,names, metric_name_and_criteria={'val_loss':np.min,'val_dice_coef_3D':np.max}):
    files = []
    folders = []
    for root, folders, files in os.walk(path):
        break
    event_files = [i for i in files if i.find('event') == 0]
    if event_files:
        try:
            print(path)
            all_dictionaries = add_to_dictionary(path, all_dictionaries,names=names, metric_name_and_criteria=metric_name_and_criteria)
        except:
            xxx = 1
    for folder in folders:
        all_dictionaries = down_folder(os.path.join(path,folder),all_dictionaries,names,metric_name_and_criteria=metric_name_and_criteria)
    return all_dictionaries

def turn_list_dictionaries_into_one(all_dictionaries, metric_name_and_criteria={'val_loss':np.min,'val_dice_coef_3D':np.max}):
    # Get the names of all variables, we will need to make a complete list
    key_names = []
    for temp_dictionary in all_dictionaries:
        key_names += [i for i in temp_dictionary.keys() if i not in key_names and i is not 'Full_Name' and i not in metric_name_and_criteria.keys()]
    total_dictionary = {'Full_Name':[]}
    for key_name in key_names:
        total_dictionary[key_name] = []
    for metric_name in metric_name_and_criteria:
        total_dictionary[metric_name] = []
    for temp_dictionary in all_dictionaries:
        full_name = temp_dictionary['Full_Name']
        if full_name not in total_dictionary['Full_Name']:
            total_dictionary['Full_Name'].append(full_name)
            for key_name in key_names:
                if key_name in temp_dictionary:
                    i = str(float(temp_dictionary[key_name])) # Gotta lay the whole value out
                    if i[-2:] == '.0':
                        i = int(float(i))
                    else:
                        i = float(i)
                    total_dictionary[key_name].append(i)
                else:
                    total_dictionary[key_name].append('')
            for metric_name in metric_name_and_criteria:
                if metric_name in temp_dictionary:
                    total_dictionary[metric_name].append(float(temp_dictionary[metric_name]))
                else:
                    if metric_name_and_criteria[metric_name] is np.max:
                        total_dictionary[metric_name].append(-np.inf)
                    else:
                        total_dictionary[metric_name].append(np.inf)
        else:
            index = total_dictionary['Full_Name'].index(full_name)
            for metric_name in metric_name_and_criteria:
                if metric_name in temp_dictionary:
                    total_dictionary[metric_name][index] = metric_name_and_criteria[metric_name]([total_dictionary[metric_name][index], float(temp_dictionary[metric_name])])
    return total_dictionary


def create_excel_from_event(input_path=None, excel_out_path=os.path.join('.','Model_Optimization.xlsx'),
                            names=['Layers','Filters','Max_Filters','Atrous_Blocks'],
                            metric_name_and_criteria={'val_loss':np.min,'val_dice_coef_3D':np.max}):
    if input_path is None:
        return None
    all_dictionaries = down_folder(input_path, [],names=names, metric_name_and_criteria=metric_name_and_criteria)
    total_dictionary = turn_list_dictionaries_into_one(all_dictionaries, metric_name_and_criteria=metric_name_and_criteria)
    df = pd.DataFrame(total_dictionary)
    df.to_excel(excel_out_path, index=0)
    return None


def plot_from_excel(excel_path,variable_name='layers',metric_name='val_loss',log_val=True,criterias=[]):
    # criteria_base = lambda x, variable_name, value: np.asarray(list(x[variable_name].values())) == value
    # criteria = [partial(criteria_base, variable_name='layers', value=5)]
    variable_name = variable_name.lower()
    data = pd.read_excel(excel_path, engine='openpyxl').to_dict()
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
