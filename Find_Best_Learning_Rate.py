import os
from keras.optimizers import Adam

class Find_Best_Learning_Rate(object):
    def __init__(self, train_generator=None, validation_generator=None, Model_val=None, epochs=0, lr=0, upper_bound=1, scale=2,
                 out_path=os.path.join('.','Learning_rates'),metrics=['accuracy'], optimizer=Adam):
        self.train_generator = train_generator
        self.validation_generator = validation_generator
        self.epochs = epochs
        self.Model_val = Model_val
        self.out_path = out_path
        self.metrics = metrics
        self.optimizer = optimizer
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        self.run_over_learning_rates(learning_rate=lr, upper_bound=upper_bound, scale=scale)

    def run_for_learning_rate(self, learning_rate):
        optimizer = self.optimizer(lr=learning_rate)
        self.Model_val.compile(optimizer, loss='categorical_crossentropy', metrics=self.metrics)
        history = self.Model_val.fit_generator(generator=self.train_generator, workers=10, use_multiprocessing=False,
                                               max_queue_size=200,shuffle=True, epochs=self.epochs, initial_epoch=0,
                                               validation_data=self.validation_generator)

        out_path = os.path.join(self.out_path, str(learning_rate) + '.txt')
        fid = open(out_path,'w+')
        for key in history.history.keys():
            if key.find('val') == 0: # Only get the validation data
                fid.write(key + ',')
                fid.write(str(history.history[key])+'\n')
        fid.close()
        return None

    def run_over_learning_rates(self, learning_rate, upper_bound, scale=2):
        while learning_rate < upper_bound:
            self.run_for_learning_rate(learning_rate)
            learning_rate *= scale
        return None

if __name__ == '__main__':
    xxx = 1