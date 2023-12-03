# -*- coding: utf-8 -*-
"""
DreamentoConverter: This file is meant to convert several 'raw' .hyp ZMax recordings
into the corresponding .edf files.

=== Inputs ===
1. path_to_HDRecorder: path to HDRecorder.exe (specify the folder name only, e.g., C:/Program Files (x86)/Hypnodyne/ZMax)
2. filenames: path to the files to be converted, e.g., see 'filenames' below
3. destination_folders: destination folders, e.g., see 'destination_folders' below

=== Outputs ===
Converted edf files in destination folders!

source: https://github.com/dreamento/dreamento 
By Mahdad Jafarzadeh
"""
import os
import shutil
import numpy as np
import subprocess

# define path to HDRecorder.exe
path_to_HDRecorder = 'C:\\Program Files (x86)\\Hypnodyne\\ZMax\\'
os.chdir(path_to_HDRecorder)

# define files to be converted 
filenames = ['C:\\PhD\\test_DreamentoConverter\\hypno1.hyp']

# define path to converted folders
destination_folders = ['C:\\PhD\\test_DreamentoConverter\\converted\\hypno1\\']

for conv in np.arange(len(filenames)):
    
    # Copy the .hyp file to HDRecoder folder for conversion
    src_path = filenames[conv]
    current_file = filenames[conv].split('\\')[-1]
    dst_path = path_to_HDRecorder + current_file
    shutil.copy(src_path, dst_path)
                    
    # Create a batch file to run conversion syntax
    myBat = open(r'DreamentoConverter.bat','w+')
    myBat.write('HDRecorder.exe -conv '+ current_file)
    myBat.close()
    
    # run the created .bat --> conversion
    print(f'Converting the file {conv+1}/{len(filenames)}...please be patient...')
    subprocess.call(path_to_HDRecorder + 'DreamentoConverter.bat')
    print(f'{conv+1}/{len(filenames)} files have been successfully converted')
    print(f'file {src_path} converted to path {destination_folders[conv]}')
    # Copy generated folder to the desired path
    shutil.copytree(path_to_HDRecorder + 'SDConvert\\', destination_folders[conv])
    
    # Remove the .hyp and .bat files from HDRecorder folder
    os.remove(path_to_HDRecorder + 'DreamentoConverter.bat')
    os.remove(dst_path)
    
print('All files have been successfully converted!')