import realTimeAutoScoring
import numpy as np

sleepScoringModel = realTimeAutoScoring.importModel("./out_QS/train/21")

recording = np.loadtxt("path/to_data.txt", delimiter=',')

dataSamplesToAnalyzeBeginIndex = 0
dataSampleCounter = 0

predictions = []

for row in recording:
    dataSampleCounter += 1
    if row[4] > 1:
        if dataSamplesToAnalyzeBeginIndex == 0:
            dataSamplesToAnalyzeBeginIndex = dataSampleCounter

        if dataSampleCounter == dataSamplesToAnalyzeBeginIndex+30*256:
            sig = recording[dataSamplesToAnalyzeBeginIndex:dataSamplesToAnalyzeBeginIndex+30*256]
            dataSamplesToAnalyzeBeginIndex = 0
            print(f"shape of sig: {len(sig)}")
            sigRef = [col[0] for col in sig]
            sigReq = [col[1] for col in sig]
            sigRef = np.asarray(sigRef)
            sigReq = np.asarray(sigReq)
            sigRef = sigRef.reshape((1, sigRef.shape[0]))
            sigReq = sigReq.reshape((1, sigReq.shape[0]))
            print(sigRef.shape, sigReq.shape)
            modelPrediction = realTimeAutoScoring.Predict_array(output_dir="./DataiBand/output/Fp1-Fp2_filtered",
                                    args_log_file="info_ch_extract.log", filtering_status=True,
                                    lowcut=0.3, highcut=30, fs=256, signal_req=sigReq, signal_ref=sigRef, model=sleepScoringModel)
            predictions.append(modelPrediction[0])
