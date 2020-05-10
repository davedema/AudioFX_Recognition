from executables import maintest, maintrain, main_traintest
from executables.modules.analysislab import user_interface

if user_interface.generate_datasets():
    main_traintest.traintest()
    maintrain.train()  # train
    maintest.test()  # test
else:
    maintrain.train()  # train
    maintest.test()  # test
