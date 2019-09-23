## If you find this code useful, please provide a reference to my github page for others www.github.com/brianmanderson , thank you!

### This is code to help find parameters to best optimize your model, it is an ongoing project ^^

For learning rates, please look at Finding_Best_Learning_Rate

Create your train and validation generators outside of this, if you're using generators found in this github they save the image files after loading (to save time), so it is best to give them to the class rather than redefine each time

Look at Plot_Best_Learning_Rates.py for plotting your generated values

        from Finding_Optimization_Parameters import LR_Finder
        import os
        
        train_generator = some_generator
        Model_val = some_model
        
        lower_lr = 1e-8
        high_lr = 1e-2
        out_path = os.path.join('.','Learning_rates','Model_Desc')
        LR_Finder.LearningRateFinder(model=Model_val,metrics=['accuracy'], out_path=out_path,
        train_generator=train_generator,lower_lr=lower_lr,high_lr=high_lr)
        
        # After the model has been made
        LR_Finder.make_plot(out_path,metric_list=['loss','accuracy'],save_path=out_path)
