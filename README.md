## If you find this code useful, please provide a reference to my github page for others www.github.com/brianmanderson , thank you!

### This is code to help find parameters to best optimize your model, it is an ongoing project ^^

### The smoothing for outputs and samplesizes comes from https://www.pyimagesearch.com/2019/08/05/keras-learning-rate-finder/ 

Create your train generators outside of this, if you're using generators found in my github they save the image files after loading (to save time), so it is best to give them to the class rather than redefine each time


        from Finding_Optimization_Parameters import LR_Finder
        import os
        
        train_generator = some_generator
        Model_val = some_model
        
        lower_lr = 1e-8
        high_lr = 1e-2
        out_path = os.path.join('.','Learning_rates','Model_Desc')
        for iteration in range(3):
                write_path = os.path.join(out_path,'Iteration_' + str(iteration))
                LR_Finder.LearningRateFinder(model=Model_val,metrics=['accuracy'], out_path=write_path,
                train_generator=train_generator,lower_lr=lower_lr,high_lr=high_lr)
        
        # After the model has been made
        iteration_paths = [os.path.join(out_path,i) for i in os.listdir(out_path)] # Providing a list will create an average output
        LR_Finder.make_plot(iteration_paths,metric_list=['loss','accuracy'],save_path=out_path, plot=True)
