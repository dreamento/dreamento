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
8. Post-processing tool (OfflineDreamento.py): Integration of all the collected data for post-processing!

## How to install?
Create all the required packages on a virtual environment:
```
conda env create --name dreamento --file dreamento.yml
```

## How to use?
Simply activate the environment you made in the previous step and run the latest version of the app:
```
conda activate dreamento
python mainwindow.py
```
Enjoy the GUI!

## Demo:
### The real-time Dreamento GUI
<img width="617" alt="Untitled" src="https://user-images.githubusercontent.com/48684369/174683169-44503f1e-2064-40a1-aa48-31ba9882fee4.png">

### The offline Dreamento GUI


### FAQ:
*1. What sources of informaion we can collect?*
Basically whatever that the Zmax Hypnodyne wearable can collect in addition to other information regarding stimulation, annotations assignment, and autoscoring, e.g., stimulation properties (color of light, intensity, etc), exact time (accurate up to the number of sample), autoscoring (real-time scored data output)

*2. How to post-process the data*  

```
conda activate dreamento
python mainwindow.py
```
Enjoy the GUI!


**CITATION:**
hello world!

Please note that this program is provided with no warranty of any kind.

**CopyRight (2021 - 22): Mahdad Jafarzadeh Esfahani, Amir Hossein Daraie, ** 

