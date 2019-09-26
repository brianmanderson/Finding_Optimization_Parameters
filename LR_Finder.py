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
                 high_lr=1e1,epochs=5,batchsize=None,steps_per_epoch=None,samplesize=2048,
                 loss = 'categorical_crossentropy', out_path=os.path.join('.','Learning_rates')):
        # store the model, stop factor, and beta value (for computing
        # a smoothed, average loss)
        if batchsize is None:
            x,y = train_generator.__getitem__(0)
            if type(x) is list:
                x = x[0]
            batchsize = x.shape[0]
        optimizer = optimizer(lr=lower_lr) # Doesn't really matter, will be over-written anyway
        self.start_lr = lower_lr
        self.stop_lr = high_lr
        model.compile(optimizer, loss=loss, metrics=metrics)
        self.model = model
        self.stopFactor = stopFactor
        self.beta = beta
        self.output_dict = {}

        if not os.path.exists(out_path):
            os.makedirs(out_path)
        self.out_path = out_path
        self.run(train_generator, batchsize=batchsize, epochs=epochs, steps_per_epoch=steps_per_epoch, samplesize=samplesize)

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

    def run(self, train_generator, batchsize=1, epochs=5, steps_per_epoch=None, samplesize=2048):
        if steps_per_epoch is None:
            steps_per_epoch = np.ceil(len(train_generator) / float(batchsize))

        if epochs is None:
            epochs = int(np.ceil(samplesize / float(steps_per_epoch)))

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
    beta = 0.98
    smooth_vals = []
    lrs = []
    for i in range(len(data_dict[metric])):
        loss = data_dict[metric][i]
        avgLoss = (beta * avgLoss) + ((1 - beta) * loss)
        smooth = avgLoss / (1 - (beta ** (i+1)))
        smooth_vals.append(smooth)
        lrs.append(data_dict['learning_rate'][i])
    return lrs, smooth_vals

def make_plot(path, metric_list=['loss'], title='', save_path=None, smooth=True, plot=False):
    if type(metric_list) != list:
        metric_list = [metric_list]
    output_pickle = [i for i in os.listdir(path) if i.find('Output.pkl') != -1]
    if output_pickle:
        data_dict = load_obj(os.path.join(path, output_pickle[0]))
        for metric in metric_list:
            if smooth:
                lrs, metrics = smooth_values(data_dict,metric=metric)
            else:
                lrs, metrics = data_dict['learning_rate'], data_dict[metric]
            plt.figure()
            plt.plot(lrs,metrics)
            plt.ylabel(metric)
            plt.xscale('log')
            plt.title(metric + ' vs Learning Rate')
            plt.xlabel('Learning Rate')
            if save_path is not None:
                out_file_name = os.path.join(save_path,title + metric+'.png')
                plt.savefig(out_file_name)
            if plot:
                plt.show()
    else:
        print('No files at ' + path)
    return None


if __name__ == '__main__':
    xxx = 1