# Dreamento: reminding you to become lucid!

## Overview

Dreamento (DREAM ENgineering TOolbox) is a an open-source Python package to record, monitor, analyze, and modulate sleep in **real-time**. The developers have done their best to build it in a modular and open-source fashion, such that other researchers can add their own features to it and extend it further. 
For any use, please cite our preprint article: https://doi.org/10.48550/arXiv.2207.03977

## Real-time features:
1. Open source!
2. Graphical user interface (GUI)
3. Data navigation and monitoring: Real-time representation of the EEG channels with adjustable time axis and amplitude scales
4. Analysis: Real-time spectrogram and peridogoram analysis
5. Autoscoring: Real-time sleep staging, open to any algorithm  (is not ideal yet, still under development)
6. Modulation: sleep modulation by visual, auditory, and tactile stimulation
7. annotations: The experimenter can set various annotations throughout the experiment.

## Post-processing:
OfflineDreamento.py: Integration of all the collected data for post-processing!

*N.B. Dreamento can now be used to analyze the recordings by HDRecorder only as well! So, any ZMax user can use Dreamento for data representation, time-freq representation, and autoscoring.*

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
- Demo on post-processing: [LINK](https://youtu.be/NzDdLlAd_F8)
- Load the relavant data files (Loading the Hypndoyne recording, Dreamento data (.txt), and annotations (.json) are mandatory, whereas EMG integration is optional (choose the relavant option form the "Plot EMG" checkbox on right of the GUI).

## Automatic sleep scoring (autoscoring):
We have recently introduced *DreamentoScorer* which is an open-source autoscoring algorithm that comes with Dreamento package. The current version of DreamentoScorer is alpha, as it is trained on ~35 nights of 
a single citizen neuroscientist only. Nevertheless, we are working hard to improve its generalizability by adding around 100 new data overall from more than 30 people! 

DreamentoScorer is a machine-learning based alogorithm which exctracts several linear and non-linear features in time and time-frequency domain from each 30-second epoch of data. The classifier is based on the LightGBM and
 we plan to add other classifiers such that the user can make a consensus of different scoring algorithms. So, stay tuned for the upcoming updates!

## Demo:
### The real-time Dreamento GUI
![Dreamento screenshot](https://user-images.githubusercontent.com/48684369/181081825-84c69c04-5ab1-4e4e-a708-9f4d59b5fb1c.png)

### The offline Dreamento GUI 
#### 1. With additional EMG measurement

![OfflineDreamento_withEMG](https://user-images.githubusercontent.com/48684369/181077650-1ce3938c-b015-4d3f-a6e1-7346f5b1046a.png)

#### 2. With no additional EMG
![OfflineDreamento_noEMG](https://user-images.githubusercontent.com/48684369/177753749-0a9b27d6-5586-4e4b-84e4-8a2284c14807.png)

### EEG - EMG synchronization
![EEG_EMG_synchronization](https://user-images.githubusercontent.com/48684369/181077226-31550c51-615f-486c-8b4f-1e5c55d8a20c.png)


### Documentation
The documentation of Dreamento can be found at: https://dreamento.github.io/docs/

### FAQ:
*1. What sources of informaion we can collect?*

Basically whatever that the ZMax Hypnodyne wearable can collect in addition to other information regarding stimulation, annotations assignment, and autoscoring, e.g., stimulation properties (color of light, intensity, etc), exact time (accurate up to the number of sample), autoscoring (real-time scored data output)

*2. How to post-process the data?*  

```
conda env create --name offlinedreamento --file offlinedreamento.yml
conda activate offlinedreamento
python OfflineDreamento.py
```
*3. Where can I find the list of dependencies?*

The *.yml files include the dependencies of real-time and offline Dreamento.

---------------------------------------------

Please note that this program is provided with no warranty of any kind.

**CITATION:**
*Jafarzadeh Esfahani, M., Daraie, A. H., Weber, F. D., & Dresler, M. (2022). Dreamento: an open-source dream engineering toolbox for sleep EEG wearables. arXiv e-prints, arXiv-2207.
https://doi.org/10.48550/arXiv.2207.03977*


**CopyRight (2021 - 22): Mahdad Jafarzadeh Esfahani, Amir Hossein Daraie** 

