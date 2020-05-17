# AudioFX_Recognition

User can try the software with the following databases:

[TRAIN SET DATABASE](https://drive.google.com/open?id=1O-mknCcecjtjRaeVAByxE91e5WFRzXLq)

[TEST SET DATABASE](https://drive.google.com/open?id=1jKyQA0UR4X2FsTq4ugXZaM8vCet6dPoG).

Otherwise user can download files from [IDMT database](https://www.idmt.fraunhofer.de/en/business_units/m2d/smt/audio_effects.html) in order to build his own train and test databases. Whether generate_datasets() is set to True, only the **train folder** in /environment/databases has to be filled.

It's mandatory to put a directory named as the class inside databases train and test folders for each class to be classified. 

The code automatically adapts itself to the number of classes and features computed by checking featuresnames() and classes() in the user_interface. There is no limit to the number of classes and feature to feed to the machine learning algorithm provided that analisyslab modules are correctly set.

The code is organized in modules and all scripts in the root directory can be run. For further insights see /pdf/report.pdf.

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
- **do_plot**  = boolean defining wheter to plot train features
- **generate_datasets**  = boolean defining wheter generate test/train datasets from a single dataset
- **test_size**  = number between 0 and 1 defining the test set length with respect to the single database length, it plays a role is generate_datasets is true
- **amplitude_scale** = set the maximum value of all audio files


