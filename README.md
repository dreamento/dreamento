# Dreamento: reminding you to become lucid!

## Overview

Dreamento (DREAM ENgineering TOolbox) is a an open-source Python package for (1) recording, monitoring, analyzing, and modulating sleep data online in **real-time** in addition to (2) **offline** post-processing the acquired data.  
- The developers have done their best to build it in a modular and open-source fashion, such that other researchers can add their own features to it and extend it further. 
- **For any use, please cite our preprint article: https://doi.org/10.48550/arXiv.2207.03977** (see also citation section in the bottom of page)

## Online (real-time) vs Offline Dreamento
- **Online/real-time Dreamento** is meant to be used for data collection
- **offline/post-processing Dreamento** is capable of analyzing both the acquired data.
- N.B. *you can analyze any ZMax data with Dreamento even if you have not recorded it via Dreamento, for instance if you recorded on the sd card or via HDRecorder! see **Post-processing** section for details)!*

### Online Dreamento:
![Dreamento screenshot](https://user-images.githubusercontent.com/48684369/181081825-84c69c04-5ab1-4e4e-a708-9f4d59b5fb1c.png)

### Offline Dreamento:
![OfflineDreamentoScreenshot](https://user-images.githubusercontent.com/48684369/212293402-de503bb8-121f-4deb-a121-595380119315.png)


## Watch Dreamento overview on YouTube:
To have a complete overview of the Dreamento package, you can watch the following episode of the tech for dreaming: [link](https://www.youtube.com/watch?v=ev78rlclxrI&ab_channel=TechforDreaming)

## Real-time features:
1. **Open source!**
2. **Graphical user interface (GUI)**
3. **Data navigation and monitoring**: Real-time representation of the EEG channels with adjustable time axis and amplitude scales
4. **Analysis**: Real-time spectrogram and peridogoram analysis
5. **Autoscoring**: Real-time sleep staging, open to any algorithm  (is not ideal yet, still under development)
6. **Modulation**: sleep modulation by visual, auditory, and tactile stimulation
7. **annotations**: The experimenter can set various annotations throughout the experiment.

## Post-processing features:

- **OfflineDreamento.py:** Integration of all the collected data for post-processing, ZMax data, annotations, TFR, **autoscroing**, and integration with other measurement modalities!
- Recently, you can also do **bulk autoscoring** with Dreamento!

## Installation and Prerequisities: 
- Download and install Anaconda (https://www.anaconda.com/).
- Download and install [the ZMax Hypnodyne software](https://hypnodynecorp.com/downloads.php).
- Download and install Dreamento (see section below).

## How to install Dreamento?
A complete **tutorial on how to install Dreamento** can be found [HERE](https://youtu.be/bDRXnMZEIyI).

- Open anaconda prompt.

- Create all the required packages on a virtual environment with the following syntax (only at the first use):
```
conda env create --name dreamento --file dreamento.yml
```

- N.B. for those who are not able to use anaconda, the software package can be installled through ```pip install -r requirements_dreamento.txt```
## How to start a recording?
A complete **tutorial on how to run Dreamento** can be found [HERE](https://youtu.be/vpmh_LiOjdw).

When you have Dreamento and Hypndoyne software installed (see sections above), follow these steps:
1. Connect the USB dongle to your pc.
2. Run HDServer.exe
3. Run HDRecorder.exe and click on "connect".
4. Open Anaconda prompt, change directory to where you installed Dreamento, activate the virtual environment you made in the previous step and then run Dreamento (mainWindow.py):
```
cd directory/to/Dreamento
conda activate dreamento
python mainwindow.py
```
5. When Dreamento's GUI started, click on "connect".
6. By clicking on the "record" button,  the recording will be started!

**Enjoy the GUI!**

## Post-processing:
- Make sure the post-processing virtual environment is installed on your pc [(video tutorial)](https://youtu.be/dpnUeIM0XDQ):
```
conda env create --name offlinedreamento --file offlinedreamento.yml
```
- Open Anaconda prompt and change the directory to where you installed Dreamento:
```
cd directory/to/Dreamento
conda activate offlinedreamento
python OfflineDreamento.py
```
You can post-process your recordings with Dreamento in three cases: (1) While having (Dreamento + Hypnodyne HDRecorder + data with other measurement modality, e.g., EMG), (2) Dreamento + Hypnodyne HDRecorder **WITHOUT** havign parallel recording with other measurement modality, and (3) recordings by ZMax only (e.g., online via HDRecorder or offline by pushing the record button on the headband).

1. Demo on post-processing (Dreamento + Hypnodyne HDRecorder + EMG data): [LINK](https://youtu.be/NzDdLlAd_F8)
2. Demo on post-processing (Dreamento + Hypnodyne HDRecorder + **WITHOUT** EMG data): the same as [LINK](https://youtu.be/NzDdLlAd_F8) , but uncheck the **Plot EMG** checkbox!
3. Demo on post-processing (** Hypnodyne HDRecorder only**): [LINK](https://youtu.be/uv6-D57b97I)

## Automatic sleep scoring (autoscoring):
We have recently introduced *DreamentoScorer* which is an open-source autoscoring algorithm that comes with Dreamento package.
DreamentoScorer is a machine-learning based alogorithm which exctracts several linear and non-linear features in time and time-frequency domain from each 30-second epoch of data. The classifier is based on the LightGBM and
 we plan to add other classifiers such that the user can make a consensus of different scoring algorithms. So, stay tuned for the upcoming updates!

This model is currently trained on 42 data and should have a reasonbable generalizability.

With DreamentoScorer, you can export not only the sleep stage predictions, but also the sleep metrics such as sleep efficiency, sleep onset latency, etc as a txt file.

DreamentoScorer not only provides the sleep stages, but also the level of certainty for each scoring (the probability of each sleep stage for each epoch).

*N.B 1: To have a reliable autoscoring with the current algorithm, the quality of both EEG channels should be satisfying.* 

*N.B 2: There is sometimes a confusion for the model to misclassify N1 as REM. Thus, we implemented an 'optional' post-scoring algorithm that replaces the detected REM before the first N2 detection with N1. This algorithm is activated by default. If you wish to deactivate it, you should set the 'apply_post_scoring_N1_correction' flag to 'False'.*

### Bulk data scoring:

In the recent update, we added the possibility for the user to provide a ```.txt``` file including the path to the folders in which ZMax data (both ```EEG L.edf``` and ```EEG R.edf```) are stored. This way, Dreamento autoscores the data and based on the user's preferences plots the results, store them and even generates sleep statistics such as the duration in each sleep stage, sleep efficiency, etc.

### Synchronization
You can collect ExG data using other device in parallel with ZMax and then use Dreamento to synchronize the outputs!
An example, is EMG recording in parallel with Dreamento, based on which the user can sync the data.

![EEG_EMG_synchronization](https://user-images.githubusercontent.com/48684369/181077226-31550c51-615f-486c-8b4f-1e5c55d8a20c.png)

### Documentation
The documentation of Dreamento can be found at: https://dreamento.github.io/docs/

### FAQ:

*1. Can I analyze the data I collected with ZMax headband (an not necessarily with Dreamento), such as a recording by pushing the record button on the headband using offline Dreamento?*

Yes! If you have a ZMax recording by Hypnodyne software (and consequently the .edf files) and want to analyze your results (e.g. automatically score the data) with Dreamento, have a look at watch: https://youtu.be/uv6-D57b97I.

*2. What sources of informaion we can collect with Dreamento?*

Basically whatever that the ZMax Hypnodyne wearable can collect in addition to other information regarding stimulation, annotations assignment, and autoscoring, e.g., stimulation properties (color of light, intensity, etc), exact time (accurate up to the number of sample), autoscoring (real-time scored data output)


*3. How to post-process the data?*  

```
conda env create --name offlinedreamento --file offlinedreamento.yml
conda activate offlinedreamento
python OfflineDreamento.py
```
*4. Where can I find the list of Python package dependencies?*

The *.yml files include the dependencies of real-time and offline Dreamento.

---------------------------------------------

Please note that this program is provided with no warranty of any kind.

## Citation

*Esfahani, M. J., Daraie, A. H., Weber, F. D., & Dresler, M. (2022). Dreamento: an open-source dream engineering toolbox for sleep EEG wearables. arXiv e-prints, arXiv-2207.
https://doi.org/10.48550/arXiv.2207.03977*


**CopyRight (2021 - 22): Mahdad Jafarzadeh Esfahani, Amir Hossein Daraie** 

