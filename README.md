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
    
    for c in np.arange(len(classes)):                      # for each class indentified by 0,1,2
        condition = np.mod(y_train, 3) == c                # condition on y_train that selects members of classes()[c]
        n_train = len(y_train[condition])                  # nuber of files of the class inside X_train
        train_feats = np.zeros((n_train, n_features))      # create a zeros matrix that will fill the c class of the object
        k=0

        for i in np.arange(len(y_train)):                  # checks on every element of y_train                   
            if y_train[i] == c:                            # if the i-th element is of c class
                train_feats[k,:] = X_train[i, :]           # fill k-th row of the zeros matrix with the ith row of X_train
                k = k+1                                    # increment k
        dict_train_feats[classes[c]] = train_feats         # at the end of the iteration, puts the matrix inside the object
  

    for c in np.arange(len(classes)):
        . # same structure of before
        dict_test_feats[classes[c]] = test_feats

We now have reconstructed dict_train_feats and dict_test_feats, but we need to store them:

    savetraindata.save_datasets(dict_train_feats, dict_test_feats)

## Savetraindata e Dataloader

#### Savetrainata
We find two functions in this module. They both save arrays and matrices using the dump option of numpy elements :
**savedata(dict_train_features, featurelst, feat_max, feat_min)** and **save_datasets(dict_train_feats, dict_test_feats)**.

The first one is used to save the data at the end of the training set, after features selection; 
it doesn't re-save dict_train_features in the case in which generate_datasets is True.
The second one is used to save test and train sets in the case in which generate_datasets is True.

#### Dataloader
This module defines many functions which retrieve the saved data returning numpy.load algorithm, those functions are:

**Trainselected()**  returns selected features
**featmax()**  returns maximus over the features
**featmin()** returns minimum over the features
**dict_train_features(c)** returns the c class matrix of dict_train_features (generate_datasets == true)
**dict_train_feats(c)** returns the c class matrix of dict_train_feats (generate_datasets == true)
**dict_test_feats(c)** returns the c class matrix of dict_train_feats (generate_datasets == true)
**columns_selected()** returns the columns selected on feature selection

## Maintrain

We first need to have reference of the path where ve have stored the train database from the **user_interface.py** element function datapathtrain()

    path = user_interface.datapathtrain()
    classes = user_interface.classes()
    datasets = user_interface.generate_datasets()

If generate_datasets is true, we get the train features via the dataloader, otherwise we need to do the training loop .
In any case we need to label the files with an y vector.

    if(datasets):
        dict_train_features = [dataloader.dict_train_feats(c) for c in classes ]
        X_train = [dict_train_features[c] for c in np.arange(len(classes))]
    else:
        dict_train_features = trainingloop.getdicttrainfeatures(path)  # compute train features
        X_train = [dict_train_features[c] for c in classes]
    y_train = [np.ones(X_train[i].shape[0], ) * i for i in np.arange(len(user_interface.classes()))]  # labels
 
We extract max and min in order to normalize the features. If requested by **user_interface.do_plot**, we plot the Train features.
We then process file in order to use  **featureselection.py** file functions that realizes feature selection and save the data.

    feat_max = np.max(np.concatenate((X_train[0], X_train[1], X_train[2]), axis=0), axis=0)
    feat_min = np.min(np.concatenate((X_train[0], X_train[1], X_train[2]), axis=0), axis=0)
    X_train_normalized = [(X_train[c] - feat_min) / (feat_max - feat_min)
                          for c in np.arange(len(user_interface.classes()))]  # normalized matrix
    if user_interface.do_plot():
        plotfeatures.plotfeats(X_train_normalized)  # plot train features
    X_train_mc_normalized = np.concatenate((X_train_normalized[0], X_train_normalized[1], X_train_normalized[2]),
                                           axis=0)
    y_train_mc = np.concatenate((y_train[0], y_train[1], y_train[2]), axis=0)
    featurelst = featureselection.getfeaturelist(X_train_mc_normalized, y_train_mc)  # feature selection
    savetraindata.savedata(dict_train_features, featurelst, feat_max, feat_min)  # save data

## Maintest

We first need to have reference of the path where ve have stored the test database from the **user_interface.py** element function datapathtest()


    # begin compute and select features
    path = user_interface.datapathtest()
    classes = user_interface.classes()
    datasets = user_interface.generate_datasets()

If generate_datasets is true, we get the test features via the dataloader, otherwise we need to do the test loop .
In any case we need to select the right number of columns (features), update X_test and label the files with an y vector.

    if (datasets):
        dict_test_features = [dataloader.dict_test_feats(c) for c in classes]
        X_test = [dict_test_features[c] for c in np.arange(len(classes))]
    else:
        dict_test_features = testloop.getdicttestfeatures(path)  # test features
        X_test = [dict_test_features[c] for c in classes]

    columns_selected = dataloader.colums_selected()  # positions of selected features
    X_test_selected = [X_test[i][:, columns_selected] for i in np.arange(len(user_interface.classes()))]  # selection
    y_test = [np.ones(X_test[i].shape[0], ) * i for i in np.arange(len(user_interface.classes()))]  # keys
    
We process datas in order to feed them to sklearn Support Vector Machine implementation. Notice that wheter generate_datasets is True we use different dataloader functions.    
    
    y_test_mc = np.concatenate((y_test[0], y_test[1], y_test[2]), axis=0)
    X_test_normalized = [
        ((X_test_selected[c] - dataloader.featmin()[columns_selected]) /
         (dataloader.featmax()[columns_selected] - dataloader.featmin()[columns_selected]))
        for c in np.arange(len(user_interface.classes()))]  # normalized matrix
    X_test_mc_normalized = np.concatenate((X_test_normalized[0], X_test_normalized[1], X_test_normalized[2]), axis=0)
    if datasets:
        X_train_normalized_loaded = [
            (dataloader.dict_train_feats(c) - dataloader.featmin()) /
            (dataloader.featmax() - dataloader.featmin())
            for c in user_interface.classes()]  # train features

    else:
        X_train_normalized_loaded = [
            (dataloader.dict_train_features(c) - dataloader.featmin()) /
            (dataloader.featmax() - dataloader.featmin())
        for c in user_interface.classes()]  # train features

    X_train_normalized_loaded_selected = [X_train_normalized_loaded[i][:, columns_selected]   for i in np.arange(len(classes)) ]  
    y_train_selected = [np.ones(X_train_normalized_loaded_selected[i].shape[0], ) * i         for i in np.arange(len(classes)) ]
    
We now feed the data to sklearn.SVM using our **supportvectormachines.py** file function

    y_test_predicted_mv = supportvectormachines.getpredictions(X_train_normalized_loaded_selected,y_train_selected,X_test_mc_normalized)
   
And then we print the metrics, which are values that allow the user to evaluate its classification, and a confusion matrix, that makes the user see how able is the software in the classification of files. We will see those modules in detail.

    print('\n\nMetrics:')
    metrics.get_metrics(dict_test_features)
    print('\n\nConfusion matrix:')
    confusionmatrix.compute_cm_multiclass(y_test_mc, y_test_predicted_mv)  # print confusion matrix



# Low Level Software Explanation

## Features extraction

### Frame analysis
### Audio analysis

## Training/Test Loop

## Feature Selection

## Metrics Calculation

## Support Vector Machine

## Confusion Matrix
