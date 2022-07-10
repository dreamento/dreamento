# Dreamento: reminding you to become lucid!

## Overview

Dreamento (DREAM ENgineering TOolbox) is a Python package to record, monitor, analyze, and modulate sleep in **real-time**. The developers have done their best to build it in a modular and open-source fashion, such that other researchers can add their own features to it and extend it further. 

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

## Installation and Prerequisities: 
- Download and install Anaconda (https://www.anaconda.com/).
- Download and install [the ZMax Hypnodyne software](https://hypnodynecorp.com/downloads.php).
- Download and install Dreamento (see section below).

## How to install Dreamento?
- Open anaconda prompt.

- Create all the required packages on a virtual environment with the following syntax (only at the first use):
```
conda env create --name dreamento --file dreamento.yml
```
## How to start a recording?
When you have Dreamento and Hypndoyne software installed (see sections above), follow these steps:
1. Connect the USB dongle to your pc.
2. Run HDServer.exe
3. Run HDRecorder.exe and click on "connect".
4. Open Anaconda prompt, change directory to where you installed Dreamento, activate the virtual environment you made in the previous step and run then run Dreamento (mainWindow.py):
```
cd directory/to/Dreamento
conda activate dreamento
python mainwindow.py
```
5. When Dreamento's GUI started, click on "connect".
6. By clicking on the "record" button,  the recording will be started!
**Enjoy the GUI!**

## Post-processing:
- Make sure the post-processing environment is installed on your pc:
```
conda env create --name offlinedreamento --file offlinedreamento.yml
```
- Open Anaconda prompt and change the directory to where you installed Dreamento:
```
cd directory/to/Dreamento
conda activate offlinedreamento
python OfflineDreamento.py
```
- Load the relavant data files (Loading the Hypndoyne recording, Dreamento data (.txt), and annotations (.json) are mandatory, whereas EMG integration is optional (choose the relavant option form the "Plot EMG" checkbox on right of the GUI).


## Demo:
### The real-time Dreamento GUI
![Dreamento screenshot](https://user-images.githubusercontent.com/48684369/177725424-9aca75e7-62c9-46c0-8a27-c17579fa8c47.png)

### The offline Dreamento GUI 
#### 1. With additional EMG measurement

![OfflineDreamento_withEMG](https://user-images.githubusercontent.com/48684369/177753415-a32ecb4c-a599-4212-bc03-b34b3de662dc.png)


#### 2. With no additional EMG
![OfflineDreamento_noEMG](https://user-images.githubusercontent.com/48684369/177753749-0a9b27d6-5586-4e4b-84e4-8a2284c14807.png)


### FAQ:
*1. What sources of informaion we can collect?*

Basically whatever that the ZMax Hypnodyne wearable can collect in addition to other information regarding stimulation, annotations assignment, and autoscoring, e.g., stimulation properties (color of light, intensity, etc), exact time (accurate up to the number of sample), autoscoring (real-time scored data output)

*2. How to post-process the data?*  

```
conda env create --name offlinedreamento --file offlinedreamento.yml
conda activate offlinedreamento
python OfflineDreamento.py
```

Please note that this program is provided with no warranty of any kind.

**CITATION:**
hello world!


**CopyRight (2021 - 22): Mahdad Jafarzadeh Esfahani, Amir Hossein Daraie** 

