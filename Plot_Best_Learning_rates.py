import os, pickle
from matplotlib import pyplot as plt
import numpy as np

def load_obj(path):
    if path.find('.pkl') == -1:
        path += '.pkl'
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        out = {}
        return out

def save_obj(path, obj): # Save almost anything.. dictionary, list, etc.
    if path.find('.pkl') == -1:
        path += '.pkl'
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.DEFAULT_PROTOCOL)
    return None

def make_plot(path, metric='acc', min_or_max = max, title=None, save_path=None):
    if title is None:
        title = metric
    data_dict = load_obj(os.path.join(path,'data_dict.pkl'))
    lr_files = [i for i in os.listdir(path) if i.find('.txt') != -1]
    for file in lr_files:
        lr = float(file.split('.txt')[0])
        if lr not in data_dict:
            data_dict[lr] = {}
            fid = open(os.path.join(path,file))
            for line in fid:
                fid_metric = line.split(',')[0]
                start = line.find('[')
                stop = line.find(']')
                values = line[start+1:stop].split(',')
                values = [float(i) for i in values]
                data_dict[lr][fid_metric] = values

    save_obj(os.path.join(path,'data_dict.pkl'),data_dict)
    lrs = list(data_dict.keys())
    metrics = []
    for lr in lrs:
        values = data_dict[lr][metric]
        value = min_or_max([float(i) for i in values])
        metrics.append(value)
    indexes = np.argsort(np.asarray(lrs))
    metrics = np.asarray(metrics)[indexes]
    lrs = np.asarray(lrs)[indexes]
    plt.figure()
    plt.plot(lrs,metrics)
    plt.ylabel(metric)
    plt.xscale('log')
    plt.title(title + ' vs Learning Rate')
    plt.xlabel('Learning Rate')
    if save_path is not None:
        out_file_name = os.path.join(save_path,title+'.png')
        plt.savefig(out_file_name)
    plt.show()
    return None

if __name__ == '__main__':
    xxx = 1