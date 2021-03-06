import os
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.backend import get_session

def reset_model(model):
    print('Reset model')
    session = get_session()
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'kernel_initializer'):
            model.layers[i].kernel.initializer.run(session=session)
    return model

class Find_Best_Learning_Rate(object):
    def __init__(self, train_generator=None, validation_generator=None, Model_val=None, epochs=2, learning_rate=0, upper_bound=1, scale=1.1,reset_model=False,
                 out_path=os.path.join('.','Learning_rates'),metrics=['accuracy'], optimizer=Adam, loss='categorical_crossentropy',num_workers=10, steps_per_epoch=None):
        self.steps_per_epoch = self.steps_per_epoch
        if steps_per_epoch is None:
            self.steps_per_epoch = len(train_generator)
        self.num_workers = num_workers
        self.reset_model = reset_model
        self.loss = loss
        self.train_generator = train_generator
        self.validation_generator = validation_generator
        self.epochs = epochs
        self.Model_val = Model_val
        self.out_path = out_path
        self.metrics = metrics
        self.optimizer = optimizer
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        self.run_over_learning_rates(learning_rate=learning_rate, upper_bound=upper_bound, scale=scale)

    def run_for_learning_rate(self, learning_rate):
        out_path = os.path.join(self.out_path, str(learning_rate) + '.txt')
        if os.path.exists(out_path):
            return None # Do not redo a run
        if self.reset_model:
            self.Model_val = reset_model(self.Model_val)
        optimizer = self.optimizer(lr=learning_rate)
        self.Model_val.compile(optimizer, loss=self.loss, metrics=self.metrics)
        history = self.Model_val.fit_generator(generator=self.train_generator, workers=self.num_workers, use_multiprocessing=False,
                                               max_queue_size=200,shuffle=True, epochs=self.epochs, initial_epoch=0,
                                               validation_data=self.validation_generator, steps_per_epoch=self.steps_per_epoch)
        fid = open(out_path,'w+')
        for key in history.history.keys():
            if key.find('val') == 0: # Only get the validation data
                fid.write(key + ',')
                fid.write(str(history.history[key])+'\n')
        fid.close()
        return None

    def run_over_learning_rates(self, learning_rate, upper_bound, scale):
        while learning_rate < upper_bound:
            print('Learning rate is ' + str(learning_rate))
            self.run_for_learning_rate(learning_rate)
            learning_rate *= scale
        return None

if __name__ == '__main__':
    xxx = 1