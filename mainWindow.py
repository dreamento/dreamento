from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QIcon
from PyQt5.QtMultimedia import QSound
from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
import sys
import time
from ZmaxHeadband import ZmaxHeadband
from ZmaxHeadband import ZmaxDataID
from gtts import gTTS       # for speech conversion
from datetime import datetime       # for saving files with exact time
from pydub import AudioSegment      # for reading mp3 and then converting 2 wav
from pathlib import Path
import json
import realTimeAutoScoring
from lspopt_all_in_one import spectrogram_lspopt   # for spectrogram plotting
from matplotlib.colors import Normalize   # for spectrogram plotting
import periodogram # for periodogram plotting
import pyqtgraph as pg     # for eeg plotting
import scipy.signal as ssignal  # for signal filtering 4 plotting EEG
from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)   # for spectrogram plotting
def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()
def toc(echo=True):
    import time
    if 'startTime_for_tictoc' in globals():
        if echo:
            print( "Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
        return (time.time() - startTime_for_tictoc)
    else:
        if echo:
            print("Toc: start time not set")
        return -1



class Window:
    def __init__(self):
        self.hb = None
        self.dlg = uic.loadUi("mainWindows.ui")
        self.dlg.rSlider.valueChanged.connect(self.rSliderChanged)
        self.dlg.gSlider.valueChanged.connect(self.gSliderChanged)
        self.dlg.bSlider.valueChanged.connect(self.bSliderChanged)
        self.dlg.pwmSlider.valueChanged.connect(self.pwmSliderChanged)
        self.dlg.audioBrowserButton.clicked.connect(self.AudioBrowserClicked)
        self.dlg.triggerLightButton.clicked.connect(self.triggerLightClicked)
        self.dlg.triggerSoundButton.clicked.connect(self.triggerSoundClicked)
        self.dlg.triggerText2SpeechButton.clicked.connect(self.triggerText2SpeechClicked)
        self.dlg.recordButton.clicked.connect(self.recordClicked)
        self.dlg.connectSoftwareButton.clicked.connect(self.connectSoftwareButton)
        self.dlg.textToSpeechLineEdit.textChanged.connect(self.textToSpeechLineEditChanged)
        self.dlg.markerLineEdit.textChanged.connect(self.markerLineEditChanged)
        self.recordingThread = None
        self.sleepScoringModel = None
        self.stimulationDataBase = {} # have info of all triggered stimulations
        # stimulation values
        r_def = 2
        g_def = 0
        b_def = 0
        reps_def = 5
        # timeOn_def = 0.1 # sec
        # timeOff_def = 0.3 # sec
        altEyes_def = False
        vibrate_def = False
        self.dlg.rValLabel.setText(str(r_def))
        self.dlg.gValLabel.setText(str(g_def))
        self.dlg.bValLabel.setText(str(b_def))
        self.dlg.rSlider.setValue(r_def)
        self.dlg.gSlider.setValue(g_def)
        self.dlg.bSlider.setValue(b_def)
        self.dlg.repetitionsBox.setValue(reps_def)
        self.dlg.altEyesBox.setChecked(altEyes_def)
        self.dlg.vibrationBox.setChecked(vibrate_def)
        self.audio_file_path = ".\\"
        self.audio_file_name = ""
        self.dlg.connectSoftwareButton.setStyleSheet("background-color: #008CBA; color: white;") # /* Blue */
        self.dlg.triggerLightButton.setEnabled(False)
        self.dlg.triggerSoundButton.setEnabled(False)
        self.dlg.triggerText2SpeechButton.setEnabled(False)
        self.dlg.recordButton.setEnabled(False)
        self.isRecording = False
        self.dlg.signalTypeComboBox.addItems(["EEGR", "EEGL", "TEMP","EEGR, EEGL","DX, DY, DZ","EEGR, EEGL, TEMP","EEGR, EEGL, TEMP, DX, DY, DZ"])
        self.dlg.signalTypeComboBox.setCurrentIndex(5)
        self.dlg.setMarkerButton.setEnabled(False)
        self.dlg.markerLineEdit.setEnabled(False)
        self.dlg.setMarkerButton.clicked.connect(self.setMarkerButtonPressed)
        self.dlg.appDisplay.setReadOnly(True)   # always read-only
        self.setupPredictionPanelInGUI(enabled=False)
        sleepScoring_def = True
        self.dlg.scoreSleepCheckBox.setChecked(sleepScoring_def)
        self.previousMarkerLineEditValue = ""
        self.dlg.markerLineEdit.returnPressed.connect(self.setMarkerButtonPressed)
        self.dlg.scoreSleepCheckBox.stateChanged.connect(lambda x: self.scoreSleepCheckBoxEnabled() if x else self.scoreSleepCheckBoxDisabled())
        self.firstRecording = True # first time auto scoring - important for initiating tensor model only x1 time.
        # {background-color:  #4CAF50;} /* Green */
        # {background-color:  #008CBA;} /* Blue */
        # {background-color:  #f44336;} /* Red */
        # {background-color:  #e7e7e7; color: black;} /* Gray */
        # {background-color:  #555555;} /* Black */
        self.dlg.graphWidget.setBackground('w')
        self.dlg.graphWidget.setLabel('left', "<span style=\"color:red;font-size:14px\">EEG (uV)</span>")
        self.dlg.graphWidget.setLabel('bottom', "<span style=\"color:red;font-size:14px\">Time (sec)</span>")
        self.displayedXrangeCounter = 0 # for pyqtgraph eeg plot dynamic range
        self.desiredXrange = 5   # set default (0,5) - (5,10) - (10-15) - ...
        self.desiredYrange = 60  # set default (-60,60)
        self.dlg.eegRangeX_SpinBox.setValue(self.desiredXrange)  # for pyqtgraph eeg plot dynamic range
        self.dlg.eegRangeY_SpinBox.setValue(self.desiredYrange)  # for pyqtgraph eeg plot dynamic range
        self.dlg.eegRangeX_SpinBox.valueChanged.connect(self.eegRangeX_SpinBox_valueChanged)
        self.dlg.eegRangeY_SpinBox.valueChanged.connect(self.eegRangeY_SpinBox_valueChanged)
        self.dlg.graphWidget.setXRange(0, self.desiredXrange, padding=0)
        self.dlg.graphWidget.setYRange(-self.desiredYrange, self.desiredYrange, padding=0)
        self.EEGLinePen1 = pg.mkPen(color=(100, 90, 150), width=1.5)
        self.EEGLinePen2 = pg.mkPen(color=(90, 170, 160), width=1.5)
        print(self.dlg.graphWidget)
        self.dlg.graphWidget.clear()
        t = [number / 256 for number in range(256*30)]
        self.eegLine1 = self.dlg.graphWidget.plot(t, np.random.randn(30*256), self.EEGLinePen1)
        self.eegLine2 = self.dlg.graphWidget.plot(t, np.random.randn(30*256), self.EEGLinePen2)
        ax = self.dlg.graphWidget.getAxis('bottom')
        ticks = range(0,30,1)
        ax.setTicks([[(v, str(v)) for v in ticks]])
        self.dlg.resetEEGPlotButton.clicked.connect(self.resetEEGPlotButtonPressed)
        self.sleepScoringMethods = ["CNN + LSTM", "LightGBM", "SVM"]    # CAUTION: use same words in recording thread as well!
        self.dlg.sleepScoringMethodComboBox.addItems(self.sleepScoringMethods)
        if self.dlg.scoreSleepCheckBox.isChecked():
            self.dlg.scoreSleepCheckBox.setText("Real-time Autoscoring with:")
            self.dlg.sleepScoringMethodComboBox.setEnabled(True) # enable

        else:
            self.dlg.scoreSleepCheckBox.setText("Autoscoring Disabled")
            self.dlg.sleepScoringMethodComboBox.setEnabled(False)  # disable

        self.dlg.scoreSleepCheckBox.stateChanged.connect(self.scoreSleepCheckBoxChanged)
        self.t_buffer = None    # for spectrogram
        self.Sxx_buffer = None  # for spectrogram
        self.t_raw_buffer = None    # for spectrogram
        self.f_buffer = None    # for spectrogram
        self.spectrogramUpdateCounter = 0   # for spectrogram
        self.spectrogramMplWidget = self.dlg.SpectrogramWidget    # for spectrogram
        self.periodogramMplWidget = self.dlg.PeriodogramWidget    # for periodogram
        self.EEGPlotWidget = self.dlg.graphWidget   # for EEG plotting
        self.scoring_predictions = []
        self.epochCounter = 0
        self.dlg.show()

    def rSliderChanged(self, value):
        self.dlg.rValLabel.setText(str(value))

    def gSliderChanged(self, value):
        self.dlg.gValLabel.setText(str(value))

    def bSliderChanged(self, value):
        self.dlg.bValLabel.setText(str(value))

    def pwmSliderChanged(self, value):
        val = value
        val = (val - 2) / 252 * 100
        s = f"{int(np.ceil(val))}%"
        s = s.rjust(3, '0')
        self.dlg.intensitySliderLabel.setText(s)

    def connectSoftwareButton(self):
        self.hb = ZmaxHeadband()
        if self.hb.socket is None: # HDServer is not running
            self.dlg.connectSoftwareButton.setStyleSheet("background-color: #f44336; color: white;")  # /* Red */
            self.dlg.connectSoftwareButton.setText("No server running")
            self.dlg.triggerLightButton.setEnabled(False)
            self.dlg.recordButton.setEnabled(False)

        else:
            self.dlg.connectSoftwareButton.setStyleSheet("background-color: #4CAF50; color: white;")  # /* Green */
            self.dlg.connectSoftwareButton.setText("Connected")
            self.dlg.triggerLightButton.setEnabled(True)
            self.dlg.recordButton.setEnabled(True)
            if self.dlg.scoreSleepCheckBox.isChecked():
                self.setupPredictionPanelInGUI(enabled=True)

            else:
                self.setupPredictionPanelInGUI(enabled=False)

    def AudioBrowserClicked(self):
        default_path = ".\\"
        self.audio_file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self.dlg, 'Open File', default_path, '*.wav', )
        self.audio_file_name = self.audio_file_path.split('/')[-1]
        self.dlg.audioNameLabel.setText(str(self.audio_file_name))
        self.dlg.triggerSoundButton.setEnabled(True)

    def triggerLightClicked(self):
        # print(self.dlg.offTimeSpinBox.value()*10)
        self.hb.stimulate((self.dlg.rSlider.value(), self.dlg.gSlider.value(), self.dlg.bSlider.value()),
                          (self.dlg.rSlider.value(), self.dlg.gSlider.value(), self.dlg.bSlider.value()),
                          self.dlg.pwmSlider.value(), pwm2=0, t1=int(self.dlg.onTimeSpinBox.value()*10),
                          t2=int(self.dlg.offTimeSpinBox.value()*10), reps=self.dlg.repetitionsBox.value(),
                          vib=self.dlg.vibrationBox.isChecked(), alt=self.dlg.altEyesBox.isChecked())
        if self.isRecording:
            # for saving time (sec) and sample number (from 1 to 250:260) of triggered simulation
            stimulationSampleNum, stimulationSecondNum = self.recordingThread.getCurrentSampleInformation()
            color = ""
            if (str(self.dlg.rSlider.value()) == "1" or str(self.dlg.rSlider.value()) == "2") and \
                self.dlg.gSlider.value() == 0 and self.dlg.bSlider.value() == 0:
                color = "Red"

            elif (str(self.dlg.gSlider.value()) == "1" or str(self.dlg.gSlider.value()) == "2") and \
                    self.dlg.bSlider.value() == 0 and self.dlg.rSlider.value() == 0:
                color = "Green"

            elif (str(self.dlg.bSlider.value()) == "1" or str(self.dlg.bSlider.value()) == "2") and \
                    self.dlg.gSlider.value() == 0 and self.dlg.rSlider.value() == 0:
                color = "Blue"

            else:
                color = f"mixed color {self.dlg.rSlider.value()}, {self.dlg.gSlider.value()}, {self.dlg.bSlider.value()}"

            self.stimulationDataBase[f"LIGHT sample {stimulationSampleNum}, second {stimulationSecondNum}"] = \
                f"""{color}, \
pwm: {self.dlg.pwmSlider.value()}, {0}, On: {int(self.dlg.onTimeSpinBox.value() * 10)}, \
Off: {int(self.dlg.offTimeSpinBox.value() * 10)}, Reps: {self.dlg.repetitionsBox.value()}, \
Vib: {self.dlg.vibrationBox.isChecked()}, Alt: {self.dlg.altEyesBox.isChecked()}"""
            print(self.stimulationDataBase)

    def text2speech(self, txt):
        file_path = ".\\voices\\text2speech"
        Path(f"{file_path}").mkdir(parents=True, exist_ok=True)     # ensures directory exists
        myVoice = gTTS(text=txt, lang='en', slow=False)     # text 2 speech
        now = datetime.now()    # for file name
        dt_string = now.strftime("text2speech-date-%Y-%m-%d-time-%H-%M-%S")
        myVoice.save(f"{file_path}\\{dt_string}.mp3")       # save as mp3
        sound = AudioSegment.from_mp3(f"{file_path}\\{dt_string}.mp3")      # reads saved mp3 for converting 2 wav
        sound.export(f"{file_path}\\{dt_string}.wav", format="wav")     # export wav
        return f"{file_path}\\{dt_string}.wav"    # returns output file name

    def triggerSoundClicked(self):
        if self.audio_file_name != "":
            QSound.play(self.audio_file_path)   # plays .wav audio file

            if self.isRecording:
                # for saving time (sec) and sample number (from 1 to 250:260) of triggered simulation
                stimulationSampleNum, stimulationSecondNum = self.recordingThread.getCurrentSampleInformation()
                self.stimulationDataBase[f"SOUND sample {stimulationSampleNum}, second {stimulationSecondNum}"] = \
                    f"""{self.audio_file_path}"""
                print(self.stimulationDataBase)

    def triggerText2SpeechClicked(self):
        if self.dlg.textToSpeechLineEdit.text() != "":
            try:
                voiceFile = self.text2speech(self.dlg.textToSpeechLineEdit.text())      # text2speech and save .wav
                QSound.play(voiceFile)      # plays saved .wav file, which contain speeched-voice
                if self.isRecording:  # if a recording is in progress...
                    # for saving time (sec) and sample number (from 1 to 250:260) of triggered simulation
                    stimulationSampleNum, stimulationSecondNum = self.recordingThread.getCurrentSampleInformation()
                    self.stimulationDataBase[f"TEXT2SPEECH sample {stimulationSampleNum}, second {stimulationSecondNum}"] = \
                        f"""{self.dlg.textToSpeechLineEdit.text()}"""
            except:
                print("No internet connection for playing speech")

    def textToSpeechLineEditChanged(self):
        if self.dlg.textToSpeechLineEdit.text() != "":
            self.dlg.triggerText2SpeechButton.setEnabled(True)

        else:
            self.dlg.triggerText2SpeechButton.setEnabled(False)

    def recordClicked(self):
        if self.isRecording is False:
            self.recordingThread = RecordThread()
            if self.firstRecording:
                if self.dlg.scoreSleepCheckBox.isChecked() and \
                        self.dlg.sleepScoringMethodComboBox.currentText() == "CNN + LSTM":
                    self.sleepScoringModel = realTimeAutoScoring.importModel("./out_QS/train/21")
                    self.firstRecording = False   # first scoring is already done above, now for next scorings (record btn pressed), no tf model will be initiated

            else:
                self.recordingThread.model_CNNLSTM = self.sleepScoringModel

            self.dlg.recordButton.setIcon(QIcon('.\\graphics\\record.png'))
            self.recordingThread.getSignalTypeFromUI(self.dlg.signalTypeComboBox.currentText())
            self.recordingThread.start()
            self.recordingThread.recordingProgessSignal.connect(self.updateRecordBtnText)
            self.dlg.setMarkerButton.setEnabled(True)
            self.dlg.markerLineEdit.setEnabled(True)
            self.dlg.appDisplay.setPlainText("")
            self.isRecording = True

        else:
            self.dlg.recordButton.setIcon(QIcon('.\\graphics\\record_gray.png'))
            self.recordingThread.stop()
            self.dlg.setMarkerButton.setEnabled(False)
            self.dlg.markerLineEdit.setEnabled(False)
            self.isRecording = False
            self.spectrogramUpdateCounter = 0 # reset this variable, which is for plotting spectrogram and it's sliding over time

        self.recordingThread.finished.connect(self.onRecordingFinished)
        self.recordingThread.recordingFinishedSignal.connect(self.onRecordingFinishedWriteStimulationDB)
        self.recordingThread.epochPredictionResultSignal.connect(self.displayEpochPredictionResult)
        self.recordingThread.sendEEGdata2MainWindow.connect(self.getEEG_from_thread) # sending data to mainWindow for plotting, scoring, etc.
        # self.dlg.graphWidget.clear()

    def updateRecordBtnText(self, timeInSeconds):
        m, s = divmod(timeInSeconds, 60)
        h, m = divmod(m, 60)
        self.dlg.recordButton.setText(f'{h:d}:{m:02d}:{s:02d}')

    def displayEpochPredictionResult(self, predResult, epochNum):
        stagesList = ['W', 'N1', 'N2', 'N3', 'REM', 'MOVE', 'UNK']
        stagesListColor = ['SlateBlue', 'MediumSeaGreen', 'DodgerBlue', 'Violet', 'Tomato', 'Gray', 'LightGray']
        self.dlg.appDisplay.appendHtml(f"<font style='color:{stagesListColor[predResult]};' size='4'>{epochNum:03}. {stagesList[predResult]}</font>")

    def onRecordingFinished(self):
        # when the recording is finished, this function is called
        self.dlg.recordButton.setIcon(QIcon('.\\graphics\\record_gray.png'))
        self.isRecording = False
        lastRecording = self.dlg.recordButton.text() # get last recording duration from the button text :)
        self.dlg.recordButton.setText(f'New Record - Last: {lastRecording}') # update button's text
        self.dlg.markerLineEdit.setPlaceholderText('A remarkable happening....')

    def onRecordingFinishedWriteStimulationDB(self, fileName):
        # save triggered stimulation information on disk:
        with open(f'{fileName}-markers.json', 'w') as fp:
            json.dump(self.stimulationDataBase, fp, indent=4, separators=(',', ': '))

        with open(f"{fileName}-predictions.txt", "a") as outfile:
            if self.scoring_predictions != []:
                # stagesList = ['W', 'N1', 'N2', 'N3', 'REM', 'MOVE', 'UNK']
                self.scoring_predictions.insert(0, -1) # TODO (what?!) first epoch is not predicted, therefore put -1 instead
                outfile.write("\n".join(str(item) for item in self.scoring_predictions))

    def setMarkerButtonPressed(self):
        if self.isRecording:    # if a recording is in progress...
            # for saving time (sec) and sample number (from 1 to 250:260) of triggered simulation
            stimulationSampleNum, stimulationSecondNum = self.recordingThread.getCurrentSampleInformation()
            self.stimulationDataBase[f"MARKER sample {stimulationSampleNum}, second {stimulationSecondNum}"] = \
                f"""{self.dlg.markerLineEdit.text()}"""
            self.dlg.markerLineEdit.setPlaceholderText(f'{self.dlg.markerLineEdit.text()} - SAVED!')
            self.previousMarkerLineEditValue = self.dlg.markerLineEdit.text()
            self.dlg.markerLineEdit.setText("")

    def markerLineEditChanged(self):
        if self.dlg.markerLineEdit.text() == "...":
            self.dlg.markerLineEdit.setText(self.previousMarkerLineEditValue)

    def scoreSleepCheckBoxEnabled(self):
        self.setupPredictionPanelInGUI(enabled=True)

    def scoreSleepCheckBoxDisabled(self):
        self.setupPredictionPanelInGUI(enabled=False)

    def setupPredictionPanelInGUI(self, enabled=True):
        if enabled:
            self.dlg.PredictionLabel.setStyleSheet("color:#000;")
            self.dlg.appDisplay.setDisabled(False) # enable
            self.dlg.sleepScoringMethodComboBox.setEnabled(True) # enable

        else:
            self.dlg.PredictionLabel.setStyleSheet("color:#888;")
            self.dlg.appDisplay.setDisabled(True) # disable
            self.dlg.sleepScoringMethodComboBox.setEnabled(False) # disable

    def resetEEGPlotButtonPressed(self):
        # pen = pg.mkPen(color=(255, 0, 0), width=1)
        # self.dlg.graphWidget.clear()
        # # self.dlg.graphWidget.setXRange(0, 30, padding=0)
        # # ay = self.dlg.graphWidget.getAxis('bottom')
        # # ticks = range(0, 30, 1)
        # # ay.setTicks([[(v, str(v)) for v in ticks]])
        # xrange = range(0, 30*256)
        # # xrange = [v/256 for v in xrange]
        # # self.dlg.graphWidget.setData(list(xrange), list(40*np.random.randn(30*256)))#, pen)
        # # self.dlg.graphWidget.plot(range(30*256), np.random.randn(30), pen)
        # # self.dlg.graphWidget.setYRange(-100, 100, padding=0)
        # # self.dlg.graphWidget.plot([1,2,3,4,5,6,7,8,9,10], [30,32,34,32,33,31,29,32,35,45], pen=pen)
        # t = [number / 256 for number in range(30*256)]
        # eeg = self.dlg.graphWidget.getPlotItem()
        # eeg.setData(t, np.random.randn(30*256))#, pen=pen)
        # self.dlg.graphWidget.setXRange(0, 30, padding=0)
        # rng = int(self.dlg.eegRangeY_SpinBox.value())
        # self.dlg.graphWidget.setYRange(-rng, rng, padding=0)
        # ay = self.dlg.graphWidget.getAxis('bottom')
        # ticks = range(0,30,1)

        # for y
        Y_rng = int(self.dlg.eegRangeY_SpinBox.value())
        self.dlg.graphWidget.setYRange(-Y_rng, Y_rng, padding=0)

        # for x
        self.desiredXrange = self.dlg.eegRangeX_SpinBox.value()  # sec - read from UI
        sec = int(np.floor(self.displayedXrangeCounter / 256))
        k = int(np.floor(sec / self.desiredXrange))
        xMin = self.desiredXrange * k
        xMax = self.desiredXrange * (k + 1)
        a_X = self.dlg.graphWidget.getAxis('bottom')
        ticks = range(xMin, xMax, 1)
        a_X.setTicks([[(v, str(v)) for v in ticks]])
        self.dlg.graphWidget.setXRange(xMin, xMax, padding=0)



    def getEEG_from_thread(self, eegSignal_r, eegSignal_l,
                           plot_EEG=False, plot_periodogram=False,
                           plot_spectrogram=False, sleep_scoring=True,
                           epoch_counter=0):
        self.epochCounter = epoch_counter

        if plot_EEG:
            # lowcut = 0.3
            # highcut = 40
            # nyquist_freq = 256 / 2.
            # low = lowcut / nyquist_freq
            # high = highcut / nyquist_freq
            # # Req channel
            # b, a = ssignal.butter(2, [low, high], btype='band')
            # sigR = ssignal.filtfilt(b, a, eegSignal_r)
            # sigL = ssignal.filtfilt(b, a, eegSignal_l)
            sigR = eegSignal_r
            sigL = eegSignal_l
            t = [number / 256 for number in range(len(eegSignal_r))]
            self.eegLine1.setData(t, sigR, pen=self.EEGLinePen1)
            self.eegLine2.setData(t, sigL, pen=self.EEGLinePen2)
            self.displayedXrangeCounter = len(sigL)# for plotting Xrange — number of displayed samples on screen

            # for x
            sec = int(np.floor(self.displayedXrangeCounter / 256))
            if sec % self.desiredXrange == 0:
                k = int(np.floor(sec / self.desiredXrange))
                xMin = self.desiredXrange * k
                xMax = self.desiredXrange * (k + 1)
                a_X = self.dlg.graphWidget.getAxis('bottom')
                ticks = range(xMin,xMax,1)
                a_X.setTicks([[(v, str(v)) for v in ticks]])
                self.dlg.graphWidget.setXRange(xMin, xMax, padding=0)

        if plot_periodogram or plot_spectrogram:
            if eegSignal_r is None:
                data = np.asarray(eegSignal_l)

            elif eegSignal_l is None:
                data = np.asarray(eegSignal_r)

            else:
                data = np.asarray(eegSignal_l)  # choose left one (no reason for this) because only one side is enought, while coder added l & r.

        if plot_spectrogram:

            sf = 256
            win_sec = 5
            fmin = 0.5
            fmax = 25
            trimperc = 2.5
            cmap = 'RdBu_r'  # 'Spectral_r'

            if self.spectrogramUpdateCounter == 0:
                # Calculate multi-taper spectrogram
                nperseg = int(win_sec * sf)
                f, t, Sxx = spectrogram_lspopt(data, sf, nperseg=nperseg, noverlap=0)
                Sxx = 10 * np.log10(Sxx)  # Convert uV^2 / Hz --> dB / Hz

                # Select only relevant frequencies (up to 30 Hz)
                good_freqs = np.logical_and(f >= fmin, f <= fmax)
                Sxx = Sxx[good_freqs, :]
                f = f[good_freqs]

                # fill previous buffer
                self.Sxx_buffer = np.copy(Sxx)
                self.t_raw_buffer = np.copy(t)
                self.f_buffer = np.copy(f)

                t /= 60  # Convert t to minutes

                # Normalization
                vmin, vmax = np.percentile(Sxx, [0 + trimperc, 100 - trimperc])
                norm = Normalize(vmin=vmin, vmax=vmax)

                self.spectrogramMplWidget.canvas.axes.clear()
                im = self.spectrogramMplWidget.canvas.axes.pcolormesh(t, f, Sxx, norm=norm, cmap=cmap, antialiased=True,
                                                                      shading="auto")
                self.spectrogramMplWidget.canvas.axes.set_xlim(t.min(), t.max())
                self.spectrogramMplWidget.canvas.axes.set_ylabel('Frequency [Hz]')
                self.spectrogramMplWidget.canvas.axes.set_xlabel('Time [mins]')
                self.spectrogramMplWidget.canvas.axes.set_title(
                    f'Spectrogram - Update {self.spectrogramUpdateCounter + 1}')
                self.spectrogramMplWidget.canvas.draw()

                self.periodogramMplWidget.canvas.axes.clear()
                periodogram.plotPowerSpectralDensity(figure=self.periodogramMplWidget.canvas.figure,
                                                     axis=self.periodogramMplWidget.canvas.axes,
                                                     sig=data)
                self.periodogramMplWidget.canvas.draw()

            else:
                # Calculate multi-taper spectrogram
                nperseg = int(win_sec * sf)
                f, t, Sxx = spectrogram_lspopt(data, sf, nperseg=nperseg, noverlap=0)
                Sxx = 10 * np.log10(Sxx)  # Convert uV^2 / Hz --> dB / Hz

                # Select only relevant frequencies (up to 30 Hz)
                good_freqs = np.logical_and(f >= fmin, f <= fmax)
                Sxx = Sxx[good_freqs, :]
                f = f[good_freqs]

                # Normalization
                vmin, vmax = np.percentile(Sxx, [0 + trimperc, 100 - trimperc])
                norm = Normalize(vmin=vmin, vmax=vmax)

                if self.spectrogramUpdateCounter < 3:  # <4 means 0,1,2,3
                    t += self.t_raw_buffer[-1]
                    t = np.append(self.t_raw_buffer, t)
                    Sxx = np.concatenate((self.Sxx_buffer, Sxx), axis=1)
                    self.t_raw_buffer = np.copy(t)   # fill previous buffer
                    t /= 60  # Convert t to minutes

                elif self.spectrogramUpdateCounter == 3:
                    t += self.t_raw_buffer[-1]
                    t = np.append(self.t_raw_buffer, t)
                    self.t_raw_buffer = t
                    Sxx = np.concatenate((self.Sxx_buffer, Sxx), axis=1)
                    t /= 60  # Convert t to minutes

                else:
                    t = self.t_raw_buffer
                    Sxx = np.concatenate((self.Sxx_buffer[:, Sxx.shape[1]:], Sxx), axis=1)

                # fill previous buffer
                self.Sxx_buffer = np.copy(Sxx)

                # self.spectrogramMplWidget.canvas.figure.clf()
                # self.spectrogramMplWidget.canvas.axes = self.spectrogramMplWidget.canvas.figure.add_subplot(111)
                self.spectrogramMplWidget.canvas.axes.clear()
                im = self.spectrogramMplWidget.canvas.axes.pcolormesh(t, f, Sxx, norm=norm, cmap=cmap, antialiased=True,
                                                                      shading="auto")
                self.spectrogramMplWidget.canvas.axes.set_xlim(t.min(), t.max())
                self.spectrogramMplWidget.canvas.axes.set_ylabel('Frequency [Hz]')
                self.spectrogramMplWidget.canvas.axes.set_xlabel('Time [mins]')
                self.spectrogramMplWidget.canvas.axes.set_title(
                    f'Spectrogram - Update {self.spectrogramUpdateCounter + 1}')
                # Add colorbar
                # cbar = self.spectrogramMplWidget.canvas.figure.colorbar(im, ax=self.spectrogramMplWidget.canvas.axes, shrink=0.95, fraction=0.1, aspect=25)
                # cbar.ax.set_ylabel('Log Power (dB / Hz)', rotation=270, labelpad=20)
                self.spectrogramMplWidget.canvas.draw()

            self.spectrogramUpdateCounter += 1

        if plot_periodogram:
            self.periodogramMplWidget.canvas.axes.clear()
            periodogram.plotPowerSpectralDensity(figure=self.periodogramMplWidget.canvas.figure,
                                                 axis=self.periodogramMplWidget.canvas.axes,
                                                 sig=data)
            self.periodogramMplWidget.canvas.draw()

        if sleep_scoring:
            algorithm = self.dlg.sleepScoringMethodComboBox.currentText()
            if self.dlg.scoreSleepCheckBox.isChecked():
                if algorithm == "CNN + LSTM":
                    if self.sleepScoringModel is not None:
                        # 30 seconds, each 256 samples... send recording for last 30 seconds to model for prediction
                        # print("[WHILE TRUE] 30 seconds data received, now analyze it")
                        sigRef = np.asarray(eegSignal_r)
                        sigReq = np.asarray(eegSignal_l)
                        sigRef = sigRef.reshape((1, sigRef.shape[0]))
                        sigReq = sigReq.reshape((1, sigReq.shape[0]))
                        modelPrediction = realTimeAutoScoring.Predict_array(
                            output_dir="./DataiBand/output/Fp1-Fp2_filtered",
                            args_log_file="info_ch_extract.log", filtering_status=True,
                            lowcut=0.3, highcut=30, fs=256, signal_req=sigReq, signal_ref=sigRef, model=self.sleepScoringModel)
                        # print(f"Model prediction is {int(modelPrediction[0])}")
                        self.displayEpochPredictionResult(int(modelPrediction[0]), int(self.epochCounter)) # display prediction result on mainWindow
                        self.scoring_predictions.append(int(modelPrediction[0]))

                elif algorithm == "LightGBM":
                    # 30 seconds, each 256 samples... send recording for last 30 seconds to model for prediction
                    # print("[WHILE TRUE] 30 seconds data received, now analyze it")
                    sigRef = np.asarray(eegSignal_r)
                    sigReq = np.asarray(eegSignal_l)
                    sigRef = sigRef.reshape((1, sigRef.shape[0]))
                    sigReq = sigReq.reshape((1, sigReq.shape[0]))
                    print(f"Model prediction is {int(3)}")  # a dummy 3
                    self.displayEpochPredictionResult(int(3),
                                                      int(self.epochCounter))  # display prediction result on mainWindow
                    self.scoring_predictions.append(int(3))

                elif algorithm == "SVM":
                    # 30 seconds, each 256 samples... send recording for last 30 seconds to model for prediction
                    # print("[WHILE TRUE] 30 seconds data received, now analyze it")
                    sigRef = np.asarray(eegSignal_r)
                    sigReq = np.asarray(eegSignal_l)
                    sigRef = sigRef.reshape((1, sigRef.shape[0]))
                    sigReq = sigReq.reshape((1, sigReq.shape[0]))
                    print(f"Model prediction is {int(4)}")  # a dummy 4
                    self.displayEpochPredictionResult(int(4),
                                                      int(self.epochCounter))  # display prediction result on mainWindow
                    self.scoring_predictions.append(int(4))

                # if int(modelPrediction[0]) == 0:
                #     self.triggerLightClicked()

    def scoreSleepCheckBoxChanged(self):
        if self.dlg.scoreSleepCheckBox.isChecked():
            self.dlg.scoreSleepCheckBox.setText("Real-time Autoscoring with:")

        else:
            self.dlg.scoreSleepCheckBox.setText("No Real-time Autoscoring")

    def eegRangeY_SpinBox_valueChanged(self, val):
        # to change range automatically with change of spin box
        print(int(val))
        if int(val)%5 == 0:
            print(int(val))
            Y_rng = int(val)
            self.dlg.graphWidget.setYRange(-Y_rng, Y_rng, padding=0)

    def eegRangeX_SpinBox_valueChanged(self, val):
        # to change range automatically with change of spin box
        if int(val) % 5 == 0:
            sec = int(np.floor(self.displayedXrangeCounter / 256))
            k = int(np.floor(sec / self.desiredXrange))
            xMin = val * k
            xMax = val * (k + 1)
            a_X = self.dlg.graphWidget.getAxis('bottom')
            ticks = range(xMin, xMax, 1)
            a_X.setTicks([[(v, str(v)) for v in ticks]])
            self.dlg.graphWidget.setXRange(xMin, xMax, padding=0)

class RecordThread(QThread):
    recordingProgessSignal = pyqtSignal(int)    # a sending signal to mainWindow - sends time info of ongoing recording to mainWindow
    recordingFinishedSignal = pyqtSignal(str)    # a sending signal to mainWindow - sends name of stored file to mainWindow
    epochPredictionResultSignal = pyqtSignal(int, int)
    sendEEGdata2MainWindow = pyqtSignal(object, object, bool, bool, bool, bool, int)

    def __init__(self, parent=None):
        super(RecordThread, self).__init__(parent)
        self.model_CNNLSTM = None
        self.threadactive = True
        self.signalType = [0,1,5,2,3,4] # "EEGR, EEGL, TEMP, DX, DY, DZ"
        self.stimulationType = ""
        self.secondCounter = 0
        self.dataSampleCounter = 0
        self.epochCounter = 0
        self.samples_db = []

    def getSignalTypeFromUI(self, sig_type):
        # to know which signals to record, based on user interface's choice in comboBox
        if sig_type == "EEGR":
            self.signalType = [ZmaxDataID.eegr.value]
        elif sig_type == "EEGL":
            self.signalType = [ZmaxDataID.eegl.value]
        elif sig_type == "TEMP":
            self.signalType = [ZmaxDataID.bodytemp.value]
        elif sig_type == "EEGR, EEGL":
            self.signalType = [ZmaxDataID.eegr.value,ZmaxDataID.eegl.value]
        elif sig_type == "DX, DY, DZ":
            self.signalType = [ZmaxDataID.dx.value,ZmaxDataID.dy.value,ZmaxDataID.dz.value]
        elif sig_type == "EEGR, EEGL, TEMP":
            self.signalType = [ZmaxDataID.eegr.value,ZmaxDataID.eegl.value,ZmaxDataID.bodytemp.value]
        elif sig_type == "EEGR, EEGL, TEMP, DX, DY, DZ":
            self.signalType = [ZmaxDataID.eegr.value,ZmaxDataID.eegl.value,ZmaxDataID.bodytemp.value, \
                               ZmaxDataID.dx.value,ZmaxDataID.dy.value,ZmaxDataID.dz.value]

    def getCurrentSampleInformation(self):
        return [self.dataSampleCounter, self.secondCounter]  # returns time info of stimulation, when called

    def sendEEGdata2main(self,
                         eegSigR=None, eegSigL=None,
                         plot_EEG=False, plot_periodogram=False, plot_spectrogram=False,
                         score_sleep=False):
        self.sendEEGdata2MainWindow.emit(eegSigR, eegSigL, plot_EEG, plot_periodogram, plot_spectrogram,
                                         score_sleep, self.epochCounter)

    def run(self):
        # This part of the cord RECORDS signal.
        # In each second, also calculates the sampling rate (# of samples received by program over stream)
        recording = []
        cols = self.signalType
        cols.extend([999,999]) # add two columns for sample number, sample time
        # cols = [ZmaxDataID.eegr.value, ZmaxDataID.eegl.value, ZmaxDataID.bodytemp.value, 999, 999]  # eegr, eegl, temp, sample number, sample time
        recording.append(cols)  # first row of received data is the col_id. eg: 0 => eegr
        hb = ZmaxHeadband()     # create a new client on the server, therefore we use it only for reading the stream

        now = datetime.now()  # for file name
        dt_string = now.strftime("recording-date-%Y-%m-%d-time-%H-%M-%S")
        file_path = f".\\recordings\\{dt_string}"
        file_name = f"{file_path}\\{dt_string}-complete.txt"
        Path(f"{file_path}").mkdir(parents=True, exist_ok=True)  # ensures directory exists

        actual_start_time = time.time()
        print(f'actual start time {actual_start_time}')

        buffer2analyzeIsReady = False
        dataSamplesToAnalyzeCounter = 0     # count samples, when reach 30*256, feed all to deep learning model
        dataSamplesToAnalyzeBeginIndex = 0
        self.secondCounter = 0
        self.epochCounter = 0

        sigR_accumulative = [] # accumulate 256*30 data samples and empty it afterward
        sigL_accumulative = []

        while True:
# =============================================================================
#             if int(self.epochCounter % 60) and dataSamplesToAnalyzeCounter == 0:
#                 del hb
#                 hb = ZmaxHeadband()
#                 print("New HB created after 60 epochs")
# =============================================================================

            self.dataSampleCounter = 0      # count samples in each second
            self.secondCounter += 1
            self.recordingProgessSignal.emit(self.secondCounter) # send second counter to the mainWindow (then show on button)
            # start_time = time.time()
            t_end = time.time() + 1
            # print(f'{i} start time {start_time}')
            print(f'{self.secondCounter} start')
            while time.time() < t_end:
                x = hb.read(cols[:-2])
                if x != []:
                    self.dataSampleCounter += 1

                    x.extend([self.dataSampleCounter, self.secondCounter])
                    recording.append(x)

                    if buffer2analyzeIsReady == False:
                        if self.secondCounter >= 2:  # ignore 1st second for analysis, because it is unstable
                            dataSamplesToAnalyzeCounter += 1
                            if dataSamplesToAnalyzeCounter == 1:  # x[dataSamplesToAnalyzeIDXbegin:dataSamplesToAnalyzeIDXbegin+30*256]
                                sigR_accumulative = []
                                sigL_accumulative = []
                                # dataSamplesToAnalyzeBeginIndex = len(recording) - 1
                                # print(f"dataSamplesToAnalyzeBeginIndex = {dataSamplesToAnalyzeBeginIndex}")

                            if dataSamplesToAnalyzeCounter <= 30 * 256:
                                sigR_accumulative.append(x[ZmaxDataID.eegr.value])
                                sigL_accumulative.append(x[ZmaxDataID.eegl.value])
                                if dataSamplesToAnalyzeCounter % 128 == 0:  # send EEG data for plotting to mainWindow
                                    # sig = recording[dataSamplesToAnalyzeBeginIndex:]
                                    # sigr = [col[ZmaxDataID.eegr.value] for col in sig]
                                    # sigl = [col[ZmaxDataID.eegl.value] for col in sig]
                                    self.sendEEGdata2main(eegSigR=sigR_accumulative, eegSigL=sigL_accumulative,
                                                          plot_EEG=True)

                            else:
                                # print("[WHILE 1 SEC] 30 seconds data received, now analyze it")
                                # print(f"this epoch {self.epochCounter+1}: {dataSamplesToAnalyzeBeginIndex}:{dataSamplesToAnalyzeBeginIndex + 30 * 256}")
                                # dataSamplesToAnalyzeCounter = 0
                                buffer2analyzeIsReady = True
                                self.epochCounter += 1

                else:
                    print("[] data")

            # end_time = time.time()
            # print(f'end time {end_time}')
            # print(f'end time {t_end} expected')
            # time_diff = end_time - start_time
            # minute = time_diff / 60
            # seconds = time_diff % 60
            # print(f"{minute} minute, {seconds} seconds")
            self.samples_db.append(self.dataSampleCounter)
            print(f'{self.dataSampleCounter} samples')
            if buffer2analyzeIsReady:
                # send eeg data of last 30 seconds (30*256 samples) to mainWindow for plotting (spectrogram and periodogram) and sleep scoring
                # sig = recording[dataSamplesToAnalyzeBeginIndex: dataSamplesToAnalyzeBeginIndex + 30 * 256]
                # sigr = [col[ZmaxDataID.eegr.value] for col in sig]
                # sigl = [col[ZmaxDataID.eegl.value] for col in sig]
                self.sendEEGdata2main(eegSigR=sigR_accumulative, eegSigL=sigL_accumulative,
                                      plot_periodogram=True, plot_spectrogram=True, score_sleep=True)
                dataSamplesToAnalyzeCounter = 0
                buffer2analyzeIsReady = False

            if self.threadactive is False:
                break # break the loop if record button is pressed again, recording stops

        actual_end_time = time.time()
        print(f'actual end time {actual_end_time}')
        time_diff = actual_end_time - actual_start_time
        minute = time_diff / 60
        seconds = time_diff % 60
        print(f"actual {minute} minute, {seconds} seconds")

        # print(recording.shape)
        np.savetxt(file_name, recording, delimiter=',')  # save recording as txt
        print(f"Recording saved to {file_name}")
        np.save("samples_db.npy",self.samples_db)

        self.recordingFinishedSignal.emit(f"{file_path}\\{dt_string}") # send path of recorded file to mainWindow

    def stop(self):
        self.threadactive = False
        self.wait()

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    dialog = Window()
    sys.exit(app.exec())
