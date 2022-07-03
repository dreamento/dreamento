# =============================================================================
# import realTimeAutoScoring
# import numpy as np
# import scipy.signal as ssignal
# 
# from model import TinySleepNet
# from minibatching import (iterate_minibatches,
#                           iterate_batch_seq_minibatches,
#                           iterate_batch_multiple_seq_minibatches)
# 
# sleepScoringModel = realTimeAutoScoring.importModel("./out_QS/train/21")
# 
# recording = np.loadtxt(".\\2021-08-19 Pilot 8 Saba\\recording-date-2021-08-19-time-07-26-40-complete.txt", delimiter=',')
# recordingValid = recording[recording[:,4]!=1]
# fs = 256
# T = 30
# lenEpoch = fs*T
# recordingValidtruncated = recordingValid[0:int(recordingValid.shape[0] - recordingValid.shape[0]%(fs*T)), :]
# 
# EEGR = recordingValidtruncated[:,0]
# EEGL = recordingValidtruncated[:,1]
# 
# sigReq = EEGL
# sigRef = EEGR
# 
# lowcut = 0.3
# highcut = 30
# nyquist_freq = fs / 2.
# low = lowcut / nyquist_freq
# high = highcut / nyquist_freq
# # Req channel
# b, a = ssignal.butter(3, [low, high], btype='band')
# signal_req = ssignal.filtfilt(b, a, sigReq)
# # Ref channel
# signal_ref = ssignal.filtfilt(b, a, sigRef)
# 
# sigReq2 = sigReq.reshape((1, sigReq.shape[0]))
# sigRef2 = sigRef.reshape((1, sigRef.shape[0]))
# 
# n_epochs = int(sigRef2.shape[1] / lenEpoch)
# 
# sigReq_epoched = np.reshape (sigReq2,
#                            (n_epochs, lenEpoch ), order='F')
# 
# sigRef_epoched = np.reshape (sigRef2,
#                            (n_epochs, lenEpoch ), order='F')
# 
# signals = sigReq_epoched - sigRef_epoched
# 
# signals *= 10**(-6)
# 
# x            = signals.astype(np.float32)
# labels       = np.ones((1, n_epochs)) # Create fake labels
# y            = labels.astype(np.int32)
# 
# Name = "Subject X"
# 
# # Init
# test_x = []
# 
# # Reshape the data to match the input of the model - conv2d
# x = np.squeeze(x)
# x = x[:, :, np.newaxis, np.newaxis]
# 
# # Casting
# x = x.astype(np.float32)
# y = y.astype(np.int32)
# 
# test_x.append(x)
# test_y = y
# 
# config = realTimeAutoScoring.config
# 
# preds = []
# 
# if config["model"] == "model-origin":
#     for night_idx, night_data in enumerate(zip(test_x, test_y)):
#         # Create minibatches for testing
#         night_x, night_y = night_data
#         test_minibatch_fn = iterate_batch_seq_minibatches(
#             night_x,
#             night_y,
#             batch_size=config["batch_size"],
#             seq_length=config["seq_length"],
#         )
#         # Evaluate
#         test_outs = sleepScoringModel.evaluate(test_minibatch_fn)
#         preds.extend(test_outs["test/preds"])
# else:
#     for night_idx, night_data in enumerate(zip(test_x, test_y)):
#         # Create minibatches for testing
#         night_x, night_y = night_data
#         test_minibatch_fn = iterate_batch_multiple_seq_minibatches(
#             [night_x],
#             [night_y],
#             batch_size=config["batch_size"],
#             seq_length=config["seq_length"],
#             shuffle_idx=None,
#             augment_seq=False,
#         )
#         if (config.get('augment_signal') is not None) and config['augment_signal']:
#             # Evaluate
#             test_outs = sleepScoringModel.evaluate_aug(test_minibatch_fn)
#         else:
#             # Evaluate
#             test_outs = sleepScoringModel.evaluate(test_minibatch_fn)
#         preds.extend(test_outs["test/preds"])
# 
#         # Save labels and predictions (each night of each subject)
#         save_dict = {
#             "y_true": test_outs["test/trues"],
#             "y_pred": test_outs["test/preds"],
#         }
# 
#         # Object.plot_comparative_hyp(hyp_true = s_trues, hyp_pred = test_outs["test/preds"] , sub_name = Name, s_preds = test_outs["test/preds"],\
#         #                      s_acc = 'N/A' , s_kappa = 'N/A' , s_f1_score = 'N/A' ,\
#         #                      mark_REM = 'active', write_metrics = False,
#         #                      Title = 'True Hyp_'+ Name , save_fig = False,\
#         #                      directory = "./")
#             # directory = "P:/3022033.01/FilesForMathijs/daily_zmax_autoscoring/daily_zmax_hypnograms/
# 
# # sampling_rate = 256
# # lowcut = 0.3
# # highcut = 30
# # nyquist_freq = sampling_rate / 2.
# # low = lowcut / nyquist_freq
# # high = highcut / nyquist_freq
# # # Req channel
# # b, a = ssignal.butter(3, [low, high], btype='band')
# # sigReq = ssignal.filtfilt(b, a, sigReq)
# # sigRef = ssignal.filtfilt(b, a, sigRef)
# 
# 
# 
# # signals = sigReq - sigRef
# #
# # signals = signals.reshape((1, signals.shape[0]))
# #
# # x = signals.astype(np.float32)
# #
# # data_epoched = np.reshape (signals,
# #                            (int(signals.shape[1] / lenEpoch), lenEpoch, 1, 1 ), order='F')
# #
# # # Init
# # test_x = []
# 
# 
# # dataSamplesToAnalyzeBeginIndex = 0
# # dataSampleCounter = 0
# #
# # predictions = []
# #
# # for row in recording:
# #     dataSampleCounter += 1
# #     if row[4] > 1:
# #         if dataSamplesToAnalyzeBeginIndex == 0:
# #             dataSamplesToAnalyzeBeginIndex = dataSampleCounter
# #
# #         if dataSampleCounter == dataSamplesToAnalyzeBeginIndex+30*256:
# #             sig = recording[dataSamplesToAnalyzeBeginIndex:dataSamplesToAnalyzeBeginIndex+30*256]
# #             dataSamplesToAnalyzeBeginIndex = 0
# #             print(f"shape of sig: {len(sig)}")
# #             sigRef = [col[0] for col in sig]
# #             sigReq = [col[1] for col in sig]
# #             sigRef = np.asarray(sigRef)
# #             sigReq = np.asarray(sigReq)
# #             sigRef = sigRef.reshape((1, sigRef.shape[0]))
# #             sigReq = sigReq.reshape((1, sigReq.shape[0]))
# #             print(sigRef.shape, sigReq.shape)
# #             modelPrediction = realTimeAutoScoring.Predict_array(output_dir="./DataiBand/output/Fp1-Fp2_filtered",
# #                                     args_log_file="info_ch_extract.log", filtering_status=True,
# #                                     lowcut=0.3, highcut=30, fs=256, signal_req=sigReq, signal_ref=sigRef, model=sleepScoringModel)
# #             predictions.append(modelPrediction[0])
# 
# =============================================================================
