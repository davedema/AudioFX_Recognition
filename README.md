# AudioFX_Recognition

[TRAIN SET DATABASE](https://drive.google.com/open?id=1O-mknCcecjtjRaeVAByxE91e5WFRzXLq)

[TEST SET DATABASE](https://drive.google.com/open?id=1jKyQA0UR4X2FsTq4ugXZaM8vCet6dPoG)

Set path to the database directories in /executables/modules/analysislab/user_interface.py before executing on a machine

main.py executes both training and testing. executables/maintrain.py executes only training and executables/maintest.py only testing, executables/plotfeatures.py executes plotting.

Feature selection selects 4 best scoring features which will be used in maintest.py to classify. Number of top scoring features used in classification can be modified in user_interface.py by setting a different k as return value kbest().

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
- **framefeats** = number of features cinouted for each frame of every audio file
- **datapathtest** = path to test database
- **datapathtrain** = path to train database
- **do_plot**  = boolean defining wheter to plot train features

