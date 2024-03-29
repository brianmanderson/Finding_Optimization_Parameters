'''
This is adapted from code found on https://www.pyimagesearch.com/2019/08/05/keras-learning-rate-finder/
'''
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
import tensorflow as tf
import os
import pickle
import copy
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


def save_obj(path, obj):  # Save almost anything.. dictionary, list, etc.
    if path.find('.pkl') == -1:
        path += '.pkl'
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.DEFAULT_PROTOCOL)
    return None


class LearningRateFinder(object):
    def __init__(self, model, train_generator, metrics=['accuracy'], optimizer=SGD, lower_lr=1e-10,
                 steps_per_epoch=None,
                 high_lr=1e0, epochs=5, loss='categorical_crossentropy', out_path=os.path.join('.', 'Learning_rates')):
        """
        :param model: Keras model
        :param train_generator: Keras generator
        :param metrics: Metrics to track, loss is automatically included
        :param optimizer: Keras optimizer
        :param lower_lr: low learning rate to investigate
        :param high_lr: high learning rate to investigate
        :param epochs: number of epochs to perform
        :param loss: defined loss
        :param out_path: path to create output.pkl file
        """
        if steps_per_epoch is None:
            steps_per_epoch = len(train_generator)
        self.steps_per_epoch = steps_per_epoch
        optimizer = optimizer(lr=lower_lr)  # Doesn't really matter, will be over-written anyway
        self.start_lr = lower_lr
        self.stop_lr = high_lr
        model.compile(optimizer, loss=loss, metrics=metrics)
        self.model = model
        self.output_dict = {}

        if not os.path.exists(out_path):
            os.makedirs(out_path)
        self.out_path = out_path
        self.run(train_generator, epochs=epochs)

    def on_epoch_end(self, epoch, logs):
        print('Save output_dict')
        save_obj(os.path.join(self.out_path, 'Output.pkl'), self.output_dict)

    def on_batch_end(self, batch, logs):
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

    def run(self, train_generator, epochs=5):
        self.lrMult = (self.stop_lr / self.start_lr) ** (1.0 / (epochs * self.steps_per_epoch))

        callback = LambdaCallback(on_batch_end=lambda batch, logs: self.on_batch_end(batch, logs))
        callback_epoch_end = LambdaCallback(on_epoch_end=lambda epoch, logs: self.on_epoch_end(epoch, logs))
        if type(train_generator) is not tf.keras.utils.Sequence:
            self.model.fit(train_generator, epochs=epochs, steps_per_epoch=self.steps_per_epoch,
                           callbacks=[callback, callback_epoch_end])
        else:
            self.model.fit_generator(generator=train_generator, workers=10, use_multiprocessing=False,
                                     max_queue_size=10,
                                     shuffle=True, epochs=epochs, callbacks=[callback, callback_epoch_end])
        save_obj(os.path.join(self.out_path, 'Output.pkl'), self.output_dict)


def smooth_values(loss_vals, beta=0.95):
    avg_loss = 0
    smooth_values = []
    for i in range(len(loss_vals)):
        loss = loss_vals[i]
        avg_loss = (beta * avg_loss) + ((1 - beta) * loss)
        smooth = avg_loss / (1 - (beta ** (i + 1)))
        smooth_values.append(smooth)
    return smooth_values


def smooth(scalars, weight=0.6):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


def make_plot(paths, metric_list=['loss'], title='', save_path=None, weight_smoothing=0.6, plot=False, auto_rates=False,
              plot_show=True):
    """
    :param paths: type(List or String), if list, will take the average value
    :param metric_list: type(List or String), metrics wanted to be looked at
    :param title: type(String), name of title for graph
    :param save_path: type(String), path to folder of graph creation
    :param weight_smoothing: float 0.0-1.0, fraction to weight on exponential smoothing (recommend 0.6-0.8)
    :param plot: type(Bool), plot graph or just save?
    :param auto_rates: type(Bool), write out min and max lr
    :return:
    """
    if type(metric_list) != list:
        metric_list = [metric_list]
    if type(paths) != list:
        paths = [paths]
    all_lrs = {}
    all_metrics = {}
    for metric in metric_list:
        all_lrs[metric] = []
        all_metrics[metric] = []
    for path in paths:
        output_pickle = [i for i in os.listdir(path) if i.find('Output.pkl') != -1]
        if output_pickle:
            data_dict = load_obj(os.path.join(path, output_pickle[0]))
            for metric in metric_list:
                lrs, metrics = data_dict['learning_rate'], data_dict[metric]
                all_lrs[metric].append(lrs)
                all_metrics[metric].append(metrics)
        else:
            print('No files at ' + path)
    out_dict = {}
    for metric in metric_list:
        metric_data = np.asarray(all_metrics[metric])
        if metric.lower() == 'auc':
            metric_data = 1 - metric_data
        lrs = np.asarray(all_lrs[metric])[0]
        avg_data = np.mean(metric_data, axis=0)
        avg_data[np.argwhere(np.isnan(avg_data))] = np.max(avg_data[np.argwhere(~np.isnan(avg_data))])
        min_lr, max_lr = None, None
        if weight_smoothing > 0.0:
            avg_data = smooth(scalars=avg_data, weight=weight_smoothing)
        if auto_rates:
            min_loss = np.min(avg_data)
            min_loss_index = np.where(avg_data == min_loss)[0][0]
            max_lr = lrs[min_loss_index]
            min_lr = lrs[0]
            for ii in range(min_loss_index, 0, -1):
                loss = avg_data[ii]
                if loss > min_loss * 5:  # If it is at least 50% higher, we're on the curve
                    break
            for i in range(ii, 10, -1):
                previous_average = np.average(avg_data[i - 10:i])
                current_average = np.average(avg_data[i:i + 10])
                average_change = (previous_average - current_average) / (
                        np.log(lrs[i + 10]) - np.log(lrs[i - 10])) * 100
                if average_change < 1:
                    min_lr = lrs[i]
                    break
        if plot:
            plot_data(lrs[:], avg_data, metric, title, plot_show, save_path, min_lr, max_lr)
        out_dict[metric] = {'min_lr': min_lr, 'max_lr': max_lr}
    return out_dict


def plot_data(lrs, metrics, metric, title, plot_show, save_path=None, min_lr=None, max_lr=None):
    plt.figure()
    plt.plot(lrs, metrics)
    plt.ylim(top=max(metrics[:len(metrics) // 2]) * 1.25, bottom=min(metrics))
    plt.ylabel(metric)
    plt.xscale('log')
    plt.title(metric + ' vs Learning Rate')
    plt.xlabel('Learning Rate')
    if min_lr is not None:
        plt.plot([min_lr, min_lr], [np.min(metrics), np.max(metrics)], 'r-', label='Min_LR: ' + str(min_lr))
    if max_lr is not None:
        plt.plot([max_lr, max_lr], [np.min(metrics), np.max(metrics)], 'k-', label='Max_LR: ' + str(max_lr))
    val = np.min(lrs)
    while val < np.max(lrs):
        for add in [2, 4, 6, 8]:
            plt.plot([val + val * add, val + val * add], [np.min(metrics), np.max(metrics)], 'y-')
        for add in [0, 1, 3, 5, 7, 9]:
            plt.plot([val + val * add, val + val * add], [np.min(metrics), np.max(metrics)], 'g-')
        val *= 10
    if min_lr is not None or max_lr is not None:
        plt.legend()
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        out_file_name = os.path.join(save_path, title + metric + '.png')
        plt.savefig(out_file_name)
    if plot_show:
        plt.show()
    else:
        plt.close()


if __name__ == '__main__':
    xxx = 1
