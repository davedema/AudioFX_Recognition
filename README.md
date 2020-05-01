# AudioFX_Recognition

Set path to the database directories in /executables/modules/trainingloop.py and /executables/modules/testloop.py before executing on a machine

main.py executes both training and testing. executables/maintrain.py executes only training and executables/maintest.py only testing.

Feature selection selects 4 best scoring features which will be used in maintest.py to classify. Number of top scoring features used in classification can be modified in featureselection.py by setting a different k as argument of SelectKBest().
