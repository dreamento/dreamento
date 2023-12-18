# Dreamento: an open-source DREAM ENgineering TOolbox

## Overview

Dreamento is a an **open-source** Python package for (1) recording, monitoring, analyzing, and modulating sleep data in **real-time** in addition to (2) **offline** post-processing the acquired data, both in a **graphical user interface (GUI)**..  
- The developers have done their best to build it in a modular and open-source fashion, such that other researchers can add their own features to it and extend it further. 
- **For any use, please cite our preprint article: see also citation section in the bottom of page

A complete tutorial on how to install and use Dreamento can be found in the user manual folder of this repository.

## Real-time vs Offline Dreamento
- **Real-time Dreamento** is meant to be used for data collection
![Dreamento screenshot](https://user-images.githubusercontent.com/48684369/181081825-84c69c04-5ab1-4e4e-a708-9f4d59b5fb1c.png)
- **offline/post-processing Dreamento** is capable of analyzing the acquired data.
![OfflineDreamentoScreenshot](https://user-images.githubusercontent.com/48684369/212293402-de503bb8-121f-4deb-a121-595380119315.png)
- N.B. *you can analyze any ZMax data with Dreamento even if you have not recorded it via Dreamento, for instance if you recorded on the sd card or via HDRecorder! see **Post-processing** section for details)!*

## Watch us!:
1. **Tech for dreaming**: [link](https://www.youtube.com/watch?v=ev78rlclxrI&ab_channel=TechforDreaming)
2. **Dream x Engineering** seminars 2023: [link](https://www.dropbox.com/sh/ztbsgvn85xavp6q/AAAUGBv2Lq8m55Gpvr7iD1iRa?dl=0&preview=DxE-2023-3.mp4)

## Real-time features:
1. **Data navigation and monitoring**: Real-time representation of the EEG channels with adjustable time axis and amplitude scales
2. **Analysis**: Real-time spectrogram and peridogoram analysis
3. **Autoscoring**: Real-time sleep staging, open to any algorithm (is not ideal yet, still under development)
4. **Sensory stimulation**: Sensory stimulatio using visual, auditory, and tactile stimuli
5. **annotations**: Capability of adding manual and automatic markers

## Post-processing features:

- **OfflineDreamento.py:** Integration of all the collected data for post-processing, ZMax data, annotations, TFR, **autoscroing**, and integration with other measurement modalities!
- **Bulk autoscoring**.
- Automatic detection of microstructural features of sleep, i.e., **eye movements during REM + slow-oscillation and spindle detection during non-REM** epochs.
- Compatability with BrainProducts. You can analyze, autoscore, and automatically detect events.
- **Automatic ERP representation** of the detected spindles, slow-oscillations, and eye movement events.

N.B. *the autoscoring of BrainProducts and event detections are done using validated YASA algorithms.*

Automatic slow oscillation and spindle event detection:
![spindles + SO](https://github.com/dreamento/dreamento/assets/48684369/4b315e1b-3d01-4975-9ebd-766d6b239ec4)

Automatic rapid eye movement events detection: 
![rem](https://github.com/dreamento/dreamento/assets/48684369/5432a355-66ff-4ad9-be20-c94490003250)


## Installation: 
Please note that the installation **Offline** and **real-time** are different, and thus two different virtual environments are required as follows:

1. Real-time Dreamento installation:

The **Microsoft Windows** users are highly recommended to install **real-time Dreamento** using Option 1 or 2. Other OS users should use Option 3.

**Option 1: installation through Anaconda** *(recommended)*:
- Download and install Anaconda (https://www.anaconda.com/).
- Download Dreamento repository from GitHub, e.g., ```git clone https://github.com/dreamento/dreamento.git```
- Open anaconda prompt and change directory to where Dreamento repository is located (e.g., ```cd C:/path/to/Dreament_folder/```).
- Create all the required packages on a virtual environment with the following syntax (only at the first use):
```
conda env create --name dreamento --file dreamento.yml
```

**Option 2: Using ```pip```**:
- Download Dreamento repository from GitHub, e.g., ```git clone https://github.com/dreamento/dreamento.git```
- Open command prompt and change directory to where Dreamento repository is located (e.g., ```cd C:/path/to/Dreament_folder/```).
- ```pip install -r requirements_dreamento.txt```

**Option 3: Other operating systems, such as Linux, Ubuntu, Fedora**:
- ```conda create -n dreamento python=3.6 && conda activate dreamento && pip install tensorflow==1.13.1 yasa gtts==2.2.3 pydub==0.25.1 PyQt5 && conda install -c anaconda qt==5.9.7 scikit-learn==0.24.2 pyqtgraph==0.11.0```

2. Offline Dreamento installation:

**Option 1: installation through Anaconda** *(recommended)*:
- Download and install Anaconda (https://www.anaconda.com/).
- Download Dreamento repository from GitHub, e.g., ```git clone https://github.com/dreamento/dreamento.git```
- Open command prompt and change directory to where Dreamento repository is located (e.g., ```cd C:/path/to/Dreament_folder/```).
- Create all the required packages on a virtual environment with the following syntax (only at the first use): 
```
conda env create --name offlinedreamento --file offlinedreamento.yml
```

**Option 2: Other operating system, e.g.,** ```Linux, Ubuntu```:
- ```conda create -n offlineDreamento -c conda-forge spyder yasa pywavelets```

*N.B. Please note that the functionality may slightly differ on Linux-based systems, due to the minor differences in the dependencies. If you have the option to use offlineDreamento on Microsoft Windows, that's highly recommended.*

## Always stay up to date!
Note: automatic Dreamento updater uses **git**. So, if you don't have git installed on your pc, first install it (https://git-scm.com/download/win)!

Note: This option only works if you **CLONE** the repository from Github, and not when you download the zip, i.e., ```git clone https://github.com/dreamento/dreamento.git``` in your desired path.
Dreamento is continuously being updated with new features! 
So we recommend you that everytime you want to use it, first, update the software.

How?
Simply double click on ```AutomaticDreamentoUpdate.exe```. You may have to extract it from ```AutomaticDreamentoUpdate.zip```, in the same path as other Dreamento files are located. That's it!

P.S. if you are using earlier Dreamento versions, where, the updater was not included, simply download this exe file from the GitHub and add it to the SAME DIRECTORY as your Dreamento files are located.

## How to start a recording?

When you have **real-time Dreamento** and Hypndoyne software installed, follow these steps:
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

N.B. A complete **tutorial on how to run real-time Dreamento** can be found [HERE](https://youtu.be/vpmh_LiOjdw).

## How to Analyze ZMax data (post-processing)
- Make sure you have installed **offline Dreamento** (see section Step-by-step installation guide)
- Open Anaconda prompt and change the directory to where you installed Dreamento:
```
cd directory/to/Dreamento
conda activate offlinedreamento
spyder
```
When spyder pops up, open ```offlinedreamento.py``` and proceed with the desired analysis. **If you are interested in autoscoring, we highly recommend this method. Otherwise, if you run offlinedreamento directly through command prompt you may get errors while autoscoring.
You can post-process your recordings with Dreamento in three cases: (1) While having (Dreamento + Hypnodyne HDRecorder + data with other measurement modality, e.g., EMG), (2) Dreamento + Hypnodyne HDRecorder **WITHOUT** havign parallel recording with other measurement modality, and (3) recordings by ZMax only (e.g., in real-time via HDRecorder or offline by pushing the record button on the headband) (4) BrainProducts post-processing.

## Manual sleep scoring:
Recently, we enabled the manual scoring feature in Dreamento. This feature is only active when all required files are provided to Dramento (1. EEG L.edf, 2. Dreametno recording, 3. Dreamento Annotations, 4. EMG file). This is because, we do not recommend manual scoring in the absence of EMG signal.

By pressing the "Manual scoring instructions", you get the list of the relavant keys for scoring. In Dreamento manual scoring, we also enabled the possibility to mark the predefined eye signals, e.g., LRLR.


## Automatic sleep scoring (autoscoring):
We have recently introduced *DreamentoScorer* which is an open-source autoscoring algorithm that comes with Dreamento package.
DreamentoScorer is a machine-learning based alogorithm which exctracts several linear and non-linear features in time and time-frequency domain from each 30-second epoch of data. The classifier is based on the LightGBM and
 we plan to add other classifiers such that the user can make a consensus of different scoring algorithms. So, stay tuned for the upcoming updates!

This model is currently trained on over 130 data and should have a reasonbable generalizability.

With DreamentoScorer, you can export not only the sleep stage predictions, but also the sleep metrics such as sleep efficiency, sleep onset latency, etc as a txt file.

DreamentoScorer not only provides the sleep stages, but also hypnodensity as a measure of the autoscoring certainty level (the probability of each sleep stage for each epoch).

*N.B 1: To have a reliable autoscoring with the current algorithm, the quality of both EEG channels should be satisfying. Always double-check by manual scoring through a human expert in case of doubt, especially when autoscoring data (1) from a very wide age-range (our algo is trained on young adults), (2) from participants with sleep disorders, (3) short nap data (the algo is trained based full overnight sleep).* 

*N.B 2: There is sometimes a confusion for the model to misclassify N1 as REM. This is anticipated as N1 is also the stage characterized by the lowest agreement among human raters. Thus, we implemented an 'optional' post-scoring algorithm that replaces the detected REM before the first N2 detection with N1. This algorithm is activated by default. If you wish to deactivate it, you should set the 'apply_post_scoring_N1_correction' flag to 'False'.*

### Bulk data scoring:

We added the possibility for the user to provide a ```.txt``` file including the path to the folders in which ZMax data (both ```EEG L.edf``` and ```EEG R.edf```) are stored. This way, Dreamento autoscores the data and based on the user's preferences plots the results, store them and even generates sleep statistics such as the duration in each sleep stage, sleep efficiency, etc.

## Dreamento Converter

Do you have several raw recordings from ZMax (.hyp) and now you want to convert them all at once? Then you need DreamentoConverter!

If that option is not available in your OfflineDreamento GUI, you may need to update your software. See "Always stay up to date!" section!

You need to create two .txt files, one with the path to all ```.hyp``` files that you need to convert, and the other .txt fileincluding all destination paths. 

## Import sleep scoring:
By checking the "Import sleep scoring" checkbox, the user can import up to 3 already scored files. Then, Dreamento presents all the scorings along with data and provide the agreement between scorers, if more than one scoring file is imported.

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
For any use of Dreamento please cite: 

*Jafarzadeh Esfahani, M., Daraie, A. H., Zerr, P., Weber, F. D., & Dresler, M. (2023). Dreamento: An open-source dream engineering toolbox for sleep EEG wearables. SoftwareX, 24, 101595. https://doi.org/10.1016/j.softx.2023.101595*

If you intend to cite to the validity of **DreamentoScorer** in particular, please also cite:
Jafarzadeh Esfahani, M., D. Weber, F., Boon, M., Anthes, S., Almazova, T., van Hal, M., ... & Dresler, M. (2023). Validation of the sleep EEG headband ZMax. bioRxiv, 2023-08. https://doi.org/10.1101/2023.08.18.553744

**CopyRight (2021 - 23): Mahdad Jafarzadeh Esfahani, Amir Hossein Daraie** 

