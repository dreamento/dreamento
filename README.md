# ZmaxCoDo

## Overview

Zmax Controller Donders (ZmaxCoDo) is Python package to record, monitor, analyze, and modulate sleep in **real-time**. The developers have done their best to build it in a modular and open-source fashion, such that other researchers can add their own features to it and extend it further. 

## Real-time features:
1. Open source!
2. Data navigation and monitoring: Real-time representation of the EEG channels with adjustable time axis and amplitude scales
3. Analysis: Real-time spectrogram and peridogoram analysis
4. Autoscoring: Real-time sleep staging, open to any algorithm  (is not ideal yet, still under development)
5. Modulation: sleep modulation by visual, auditory, and tactile stimulation
6. annotations: The experimenter can set various annotations throughout the experiment.

## How to install?
Create all the required packages on a virtual environment:
```
conda env create --name zmax --file zmax.yml
```

## How to use?
Simply activate the environment you made in the previous step and run the latest version of the app:
```
conda activate zmax
python mainwindow.py
```

## Demo:
### Main user interface (UI)
<img width="617" alt="Untitled" src="https://user-images.githubusercontent.com/48684369/174683169-44503f1e-2064-40a1-aa48-31ba9882fee4.png">

### FAQ:
*1. What sources of informaion we can collect?*
Basically whatever that the Zmax Hypnodyne can collect in addition to other information regarding stimulation, annotations assignment, and autoscoring, e.g., stimulation properties (color of light, intensity, etc), exact time (accurate up to the number of sample), autoscoring (real-time scored data output)

*2. How to post-process the data*  
For post-processing of the data collected by ZmaxCoDo, one should use: https://github.com/ZmaxDonders/ZmaxCoDoAnalyzer

**CITATION:**
hello world!

**CopyRight (2021 - 22): Amir Hossein Daraie, Mahdad Jafarzadeh Esfahani** 
