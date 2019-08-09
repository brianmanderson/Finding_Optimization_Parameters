## If you find this code useful, please provide a reference to my github page for others www.github.com/brianmanderson , thank you!

### This is code to help find parameters to best optimize your model, it is an ongoing project ^^

For learning rates, please look at Finding_Best_Learning_Rate

Create your train and validation generators outside of this, if you're using generators found in this github they save the image files after loading (to save time), so it is best to give them to the class rather than redefine each time

Look at Plot_Best_Learning_Rates.py for plotting your generated values

        from Finding_Optimization_Parameters.Find_Best_Learning_Rate import Find_Best_Learning_Rate
        from keras.optimizers import Adam
        import os
        
        train_generator = some_generator
        validation_generator = some_generator
        Model_val = some_model
        
        Find_Best_Learning_Rate(train_generator=train_generator,validation_generator=validation_generator,
                                Model_val=Model_val,epochs=8,out_path=os.path.join('.','test'),learning_rate=1e-6, 
                                upper_bound=1e-3, metrics=['accuracy'], optimizer=Adam)
