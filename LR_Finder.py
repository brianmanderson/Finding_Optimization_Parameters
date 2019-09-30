# import the necessary packages
'''
This is adapted from code found on https://www.pyimagesearch.com/2019/08/05/keras-learning-rate-finder/
'''
from keras.callbacks import LambdaCallback
from keras.optimizers import Adam
from keras import backend as K
import os, pickle
import numpy as np
import matplotlib.pyplot as plt


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


class LearningRateFinder(object):
    def __init__(self, model, train_generator, stopFactor=4, beta=0.98,metrics=['accuracy'], optimizer=Adam, lower_lr=1e-10,
                 high_lr=1e0,epochs=5,loss = 'categorical_crossentropy', out_path=os.path.join('.','Learning_rates'), max_loss=None):
        # store the model, stop factor, and beta value (for computing
        # a smoothed, average loss)
        optimizer = optimizer(lr=lower_lr) # Doesn't really matter, will be over-written anyway
        self.start_lr = lower_lr
        self.stop_lr = high_lr
        model.compile(optimizer, loss=loss, metrics=metrics)
        self.model = model
        self.stopFactor = stopFactor
        self.beta = beta
        self.max_loss = max_loss
        self.output_dict = {}

        if not os.path.exists(out_path):
            os.makedirs(out_path)
        self.out_path = out_path
        self.run(train_generator, epochs=epochs)

    def on_epoch_end(self, epoch, logs):
        print('Save output_dict')
        save_obj(os.path.join(self.out_path, 'Output.pkl'), self.output_dict)

    def on_batch_end(self, batch, logs):
        if self.max_loss is None:
            self.max_loss = logs['loss']
        elif logs['loss'] > 2*self.max_loss:
            print('Stopping early')
            self.model.stop_training = True
        # grab the current learning rate and add log it to the list of
        # learning rates that we've tried
        lr = K.get_value(self.model.optimizer.lr)
        for key in logs:
            if key not in self.output_dict:
                self.output_dict[key] = [logs[key]]
            else:
                self.output_dict[key].append(logs[key])
        if 'learning_rate' not in self.output_dict:
            self.output_dict['learning_rate'] = [lr]
        else:
            self.output_dict['learning_rate'].append(lr)
        # increase the learning rate
        lr *= self.lrMult
        K.set_value(self.model.optimizer.lr, lr)

    def run(self, train_generator,epochs=5):
        steps_per_epoch = len(train_generator)

        self.lrMult = (self.stop_lr / self.start_lr) ** (1.0 / (epochs*steps_per_epoch))

        # construct a callback that will be called at the end of each
        # batch, enabling us to increase our learning rate as training
        # progresses
        callback = LambdaCallback(on_batch_end=lambda batch, logs: self.on_batch_end(batch, logs))
        callback_epoch_end = LambdaCallback(on_epoch_end=lambda epoch, logs: self.on_epoch_end(epoch, logs))

        # check to see if we are using a data iterator
        self.model.fit_generator(generator=train_generator, workers=10, use_multiprocessing=False,max_queue_size=200,
                                 shuffle=True, epochs=epochs, callbacks=[callback, callback_epoch_end])
        save_obj(os.path.join(self.out_path,'Output.pkl'),self.output_dict)

def smooth_values(data_dict,metric='loss'):
    avgLoss = 0
    beta = 0.95
    smooth_vals = []
    lrs = []
    for i in range(len(data_dict[metric])):
        loss = data_dict[metric][i]
        avgLoss = (beta * avgLoss) + ((1 - beta) * loss)
        smooth = avgLoss / (1 - (beta ** (i+1)))
        smooth_vals.append(smooth)
        lrs.append(data_dict['learning_rate'][i])
    return lrs, smooth_vals

def make_plot(paths, metric_list=['loss'], title='', save_path=None, smooth=True, plot=False):
    if type(metric_list) != list:
        metric_list = [metric_list]
    if type(paths) != list:
        paths = [paths]
    for metric in metric_list:
        all_lrs = {metric:[]}
        all_metrics = {metric:[]}
    for path in paths:
        output_pickle = [i for i in os.listdir(path) if i.find('Output.pkl') != -1]
        if output_pickle:
            data_dict = load_obj(os.path.join(path, output_pickle[0]))
            for metric in metric_list:
                if smooth:
                    lrs, metrics = smooth_values(data_dict,metric=metric)
                else:
                    lrs, metrics = data_dict['learning_rate'], data_dict[metric]

                all_lrs[metric].append(lrs)
                all_metrics[metric].append(metrics)
        else:
            print('No files at ' + path)

    for metric in metric_list:
        metric_data = np.asarray(all_metrics[metric])
        lrs = np.asarray(all_lrs[metric])
        averaged_data = np.mean(metric_data,axis=0)
        plot_data(lrs[0,:], averaged_data, metric, title, plot, save_path)
    return None

def plot_data(lrs, metrics, metric, title, plot, save_path=None):
    plt.figure()
    plt.plot(lrs, metrics)
    plt.ylabel(metric)
    plt.xscale('log')
    plt.title(metric + ' vs Learning Rate')
    plt.xlabel('Learning Rate')
    if save_path is not None:
        out_file_name = os.path.join(save_path, title + metric + '.png')
        plt.savefig(out_file_name)
    if plot:
        plt.show()

if __name__ == '__main__':
    xxx = 1