# AudioFX_Recognition

[TRAIN SET DATABASE](https://drive.google.com/open?id=1O-mknCcecjtjRaeVAByxE91e5WFRzXLq)

[TEST SET DATABASE](https://drive.google.com/open?id=1jKyQA0UR4X2FsTq4ugXZaM8vCet6dPoG)


The code has been modularized in order to increase performance, flexibility and fixability.

Set parameters and paths to the database directories  in /executables/modules/analysislab/user_interface.py before executing on a machine.

## User interface 

In the folder AudioFX_Recognition/executables/modules/**analysislab** there is the **user_interface.py** file.
Here the user can choose the subsequent options for the software :

- **Fs** = sampling frequency
- **winlength** = window length for windowing operations
- **hopsize** = hopsize for windowing operations
- **window** = choose the kind of [window](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html)
- **classes** = names of the classes to distinguish 
- **featuresnames** = names of the calculated features
- **kbest** = number of features chosen by feature selection
- **framefeats** = number of features computed with frameanalysis.getframefeatures()
- **datapathtest** = path to test database
- **datapathtrain** = path to train database
- **datapath** = path to database that will be used if generate_datasets is True
- **do_plot**  = boolean defining wheter to plot train features
- **generate_datasets**  = boolean defining wheter generate test/train datasets from a single dataset
- **test_size**  = number between 0 and 1 defining the test set length with respect to the single database length, it plays a role is generate_datasets is true

Those are the parameters with which the user sets the software.

# High level Software Explanation
Here we will analyse the structure and the overall architecture of the code, providing hints about the behaviour of the software at an higher level of abstraction.

## Main
Here we define the overall structure of the code, we need to differentiate between the case in which we want to generate test/train 
datasets from a single dataset and the case in which we already have our datasets. 
This is specified by the **user_interface.generate_datasets**, therefore 
### if datasets == True
 the software will implement this pipeline: 
   
    main_traintest.traintest()
    maintrain.train()  # train
    maintest.test()  # test
    
### if datasets == False
the software will implement this other traditional pipeline:
   
    maintrain.train()  # train
    maintest.test()  # test
  
## Main_traintest

## Save_Data e Dataloader

## Maintrain

We first need to have reference of the path where ve have stored the train database from the **user_interface.py** element function datapathtrain()

We extract train features via the function  getdicttrainfeatures(path) which is in the module **trainingloop.py**, there fore we obtain 
dict_train_features is an object that stores the datas in a matrix for each class
## Maintest



# Low Level Software Explanation

## Features extraction

### Frame analysis
### Audio analysis

## Training/Test Loop

## Feature Selection

## Metrics Calculation

## Support Vector Machine

## Confusion Matrix
