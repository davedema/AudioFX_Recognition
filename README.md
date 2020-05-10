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
This is specified by the **user_interface.generate_datasets**, therefore:

#### if datasets == True
 the software will implement this pipeline: 
   
    main_traintest.traintest() # splits the dataset and stores train and test results
    maintrain.train()  # processes train results (feature selection, plot)
    maintest.test()  # processes test results coherenlty to the selected columns in feature selection, 
                       therefore computes metrics, uses Supported Vector Machines, prints confusion matrix
    
#### if datasets == False
the software will implement this other traditional pipeline:
   
    maintrain.train()  # train results + processing (feature selection, plot)
    maintest.test()  # test results + processing + metrics, Support Vector Machines, confusion matrix
  
## Main_traintest

This part of the code will be executed only if user_interface.generate_datasets() is set to True, therefore when we need to generate Train/Test database from a single one.

In order to do so, one has to get the features for each file in the database. 
The features are stored via the trainingloop inside the **dict_features** object, which is indexable by the classes names.

    path = user_interface.datapath()
    dict_features = trainingloop.getdicttrainfeatures(path)  # compute train features
    classes = user_interface.classes()
    test_size = user_interface.test_size()

Now we need to split the features stored in dict_features in train and test features, therefore we concatenate thevalues for the three classes in a single data matrics X_mc and label the data of X_mc with the y_mc array

    X = [dict_features[c] for c in classes]
    y = [np.ones(X[i].shape[0], ) * i for i in np.arange(len(classes))]  # keys
    y_mc = np.concatenate((y[0], y[1], y[2]), axis=0)
    X_mc = np.concatenate((X[0], X[1], X[2]), axis=0)

We use a function of sklearn in order to randomly pick train and test sets, the parameter **test_size** is specified in the user_interface

    X_train, X_test, y_train, y_test = train_test_split(X_mc, y_mc, test_size=test_size )

We now need to rebuild the objects with class clustered data since it allows more flexibility 

    dict_train_feats = {'NoFX': [], 'Distortion': [], 'Tremolo': []}
    dict_test_feats = {'NoFX': [], 'Distortion': [], 'Tremolo': []} # I declare the empty objects 
    n_features = len(user_interface.featuresnames()) 
    
    for c in np.arange(len(classes)):                         # for each class indentified by 0,1,2
        condition = np.mod(y_train, 3) == c                   # condition on y_train that selects members of classes()[c]
        n_train = len(y_train[condition])                     # nuber of files of the class inside X_train
        train_feats = np.zeros((n_train, n_features))         # create a zeros matrix that will fill the c class of the object
        k=0

        for i in np.arange(len(y_train)):                     # checks on every element of y_train                   
            if y_train[i] == c:                               # if the i-th element is of c class
                train_feats[k,:] = X_train[i, :]              # fill k-th row of the zeros matrix with the ith row of X_train
                k = k+1                                       # increment k
        dict_train_feats[classes[c]] = train_feats            # at the end of the iteration, puts the matrix inside the object
    

    for c in np.arange(len(classes)):
        . # same structure of before
        dict_test_feats[classes[c]] = test_feats

We now have reconstructed dict_train_feats and dict_test_feats, but we need to store them:

    savetraindata.save_datasets(dict_train_feats, dict_test_feats)

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
