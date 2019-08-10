import os
from matplotlib import pyplot as plt
import numpy as np

def make_plot(path, name='acc', min_or_max = max):
    files = [i for i in os.listdir(path) if i.find('.txt') != -1]
    lrs = []
    metrics = []
    for file in files:
        lrs.append(float(file.strip('.txt')))
        fid = open(os.path.join(path,file))
        for line in fid:
            if line.find(name) != -1:
                start = line.find('[')
                stop = line.find(']')
                values = line[start+1:stop].split(',')
                value = min_or_max([float(i) for i in values])
                metrics.append(value)
                break
        fid.close()
    indexes = np.argsort(np.asarray(lrs))
    metrics = np.asarray(metrics)[indexes]
    lrs = np.asarray(lrs)[indexes]
    plt.figure()
    plt.plot(lrs,metrics)
    plt.ylabel(name)
    plt.xscale('log')
    plt.title(name + ' vs Learning Rate')
    plt.xlabel('Learning Rate')
    plt.show()
    return None

if __name__ == '__main__':
    path = r'K:\Morfeus\BMAnderson\CNN\Data\Data_Liver\Liver_Segments\Keras\3D_UNet36_Atrous_Block_no_start_conv_2\learning_rates'
    make_plot(path,name='val_dice_coef_3D',min_or_max=max)