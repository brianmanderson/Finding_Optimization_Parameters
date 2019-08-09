import os
from matplotlib import pyplot as plt
import numpy as np

def make_plot(path, name='acc'):
    files = [i for i in os.listdir(path) if i.find('.txt') != -1]
    lrs = []
    metrics = []
    for file in files:
        lrs.append(float(file.strip('.txt')))
        fid = open(os.path.join(path,file))
        for line in fid:
            if line.find(name) != 0:
                metrics.append(float(fid.readline().strip('\n').split(',')[1]))
                break
        fid.close()
    indexes = np.argsort(np.asarray(lrs))
    metrics = np.asarray(metrics)[indexes]
    lrs = np.asarray(lrs)[indexes]
    plt.figure()
    plt.plot(lrs,metrics)
    plt.ylabel('Dice')
    plt.xscale('log')
    plt.title('Dice vs Learning Rate')
    plt.xlabel('Learning Rate')
    plt.show()
    return None

if __name__ == '__main__':
    path = r'K:\Morfeus\BMAnderson\CNN\Data\Data_Liver\Data\Keras\v3_freezing\Learning_rate_info'
    make_plot(path,name='acc')