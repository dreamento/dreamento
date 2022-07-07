from yasa_imported import yasa
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import signal


dataset = np.loadtxt("path/to/data.txt", delimiter=',')

data = dataset[10*256:(10+30)*256,0]
# data =  signal.resample(data, int(len(data)/256*100))
# data = data.reshape((data.shape[0],1))

sls = yasa.SleepStaging_From_NumpyArray(data)
y_pred = sls.predict(path_to_model=".\\clf_eeg_lgb_0.4.0.joblib")
out = sls.get_features()
# times, epochs = yasa.sliding_window(data, sf=100, window=30)