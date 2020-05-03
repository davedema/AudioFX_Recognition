# AudioFX_Recognition

[TRAIN SET DATABASE](https://drive.google.com/open?id=1O-mknCcecjtjRaeVAByxE91e5WFRzXLq)

[TEST SET DATABASE](https://drive.google.com/open?id=1jKyQA0UR4X2FsTq4ugXZaM8vCet6dPoG)

Set path to the database directories in /executables/modules/trainingloop.py and /executables/modules/testloop.py before executing on a machine

main.py executes both training and testing. executables/maintrain.py executes only training and executables/maintest.py only testing.

Feature selection selects 4 best scoring features which will be used in maintest.py to classify. Number of top scoring features used in classification can be modified in featureselection.py by setting a different k as argument of Chi2().

## User interface 

in the folder AudioFX_Recognition/executables/modules/**analysislab** there is the **user_interface.py** file, here the user can choose 
the options of the software :

..*
1. **Fs** = sampling frequency
2. **winlength** = window length for windowing operations
3. **hopsize** = hopsize for windowing operations
4. **window** = choose the kind of [window](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html)
5. **classes** = names of the classes to distinguish 
6. **featuresnames** = names of the calculated features
7. **kbest** = number of features chosen by feature selection
8. **framefeats** = number of features cinouted for each frame of every audio file
9. **datapathtest** = path to test database
10. **datapathtrain** = path to train database
11. **do_plot**  = boolean defining wheter to plot train features
..*

