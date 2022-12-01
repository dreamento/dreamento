""" - Main model is based on tinysleepnet - revised by Mahdad Jafarzadeh -Feb & Mar 2021

    - This function should be used to directly predict from an Array (useful for, e.g. real-time scoring)

    - Usage:
            1) define the location, where the already trained model is located:
                    ...
                    model_dir         = "./out_iBand/train",
                    ...
            2) Assign the sampling freq. and the input array to be scored

                    fs                =  256
                    signal_req        = np.random.rand(1,fs*T)

            3) Prediction example:
                     Predict_array(output_dir      = "./DataiBand/output/Fp1-Fp2_filtered",
                     args_log_file    = "info_ch_extract.log",
                     model_dir        = "./out_iBand/train",
                     log_file         = "./out_iBand/log_file_pred.log",
                     filtering_status = True,
                     lowcut           = 0.3,
                     highcut          = 30,
                     n_folds = 1,
                     n_subjects = 1,
                     fs =  256,
                     signal_req = np.random.rand(1,7680)
                     )
    """
# %% Import libs
import time
import os
import numpy as np
import scipy.signal as ssignal
from model import TinySleepNet
from minibatching import (iterate_minibatches,
                          iterate_batch_seq_minibatches,
                          iterate_batch_multiple_seq_minibatches)
from logger import get_logger
# from ssccoorriinngg import ssccoorriinngg  # TODO: add plotting if needed

config = {
        # Train
        "n_epochs": 300,
        "learning_rate": 1e-4,
        "adam_beta_1": 0.9,
        "adam_beta_2": 0.999,
        "adam_epsilon": 1e-8,
        "clip_grad_value": 5.0,
        "evaluate_span": 50,
        "checkpoint_span": 50,

        # Early-stopping
        "no_improve_epochs": 50,

        # Model
        "model": "model-mod-8",
        "n_rnn_layers": 1,
        "n_rnn_units": 256,
        "sampling_rate": 256.0,
        "input_size": 7680,
        "n_classes": 5,
        "l2_weight_decay": 1e-3,

        # Dataset
        "dataset": "tmp",
        "data_dir": "tmp",
        "n_folds": 1,
        "n_subjects": 1,

        # Data Augmentation
        "augment_seq": True,
        "augment_signal_full": True,
        "weighted_cross_ent": True,
        "batch_size": 1,
        "seq_length": 1,
    }

def importModel(best_model_dir="./out_QS/train/4"):
    global config
    # Add dummy class weights
    config["class_weights"] = np.ones(config["n_classes"], dtype=np.float32)
    print(os.path.join(best_model_dir))
    model = TinySleepNet(
        config=config,
        output_dir=os.path.join(best_model_dir),
        use_rnn=True,
        testing=True,
        use_best=True, )
    return model


#####~~~~~~~~~~~~~~~~~~~~~~~~~~ Defining the class ~~~~~~~~~~~~~~~~~~~~~~~~####
def Predict_array(output_dir="./DataiBand/output/Fp1-Fp2_filtered",
                  args_log_file="info_ch_extract.log",
                  filtering_status=True,
                  lowcut=0.3,
                  highcut=30,
                  fs=256,
                  signal_req = np.ones((1,7680)), # Requested (EEG L)
                  signal_ref = np.ones((1,7680)), # Reference  (EEG R)
                  model=None,
                  single_epoch = True
                  ):
    # %% Initializing paths
    output_dir = output_dir

    filtering_status = filtering_status
    # %% Initialization of prediction parameters
    global config

    # Add dummy class weights
    config["class_weights"] = np.ones(config["n_classes"], dtype=np.float32)

    # Create ssccoorriinngg object
    # =============================================================================
    #     if plot_hyp == True:
    #         Object = ssccoorriinngg(filename='', channel='', fs = fs, T = 30)
    # =============================================================================

    preds = []
    # %% Output dir creation
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # %% Create logger
    logger = get_logger(args_log_file, level="info")

    logger.info("Loading new array for prediciton...")

    # Filter properties
    if filtering_status:
        lowcut = lowcut
        highcut = highcut
        nyquist_freq = fs / 2.
        low = lowcut / nyquist_freq
        high = highcut / nyquist_freq
        # Req channel
        b, a = ssignal.butter(3, [low, high], btype='band')
        signal_req = ssignal.filtfilt(b, a, signal_req)
        signal_ref = ssignal.filtfilt(b, a, signal_ref)

    x = signal_req - signal_ref

    # Reshape the data to match the input of the model - conv2d
    if single_epoch:
        x = x[[0], :, np.newaxis, np.newaxis]

    else:
        x = x[:, :, np.newaxis, np.newaxis]

    # Casting
    x = x.astype(np.float32)

    logger.info("\n=======================================\n")

    # %% Prediction section

    # Initializing the model
    logger.info("Initializing the model ...")

    # Init preds and trues per subject
    s_preds = []
    test_x = []
    # final shape of x_test and y_test
    test_x.append(x)

    # Generate fake y and y_test
    y = [1]  # dummy
    test_y = []
    test_y.append(y)

    # Print test set
    logger.info("Shape: ")
    for _x in test_x: logger.info(_x.shape)

    if config["model"] == "model-origin":

        for night_idx, night_data in enumerate(zip(test_x, test_y)):
            # Create minibatches for testing
            night_x, night_y = night_data
            test_minibatch_fn = iterate_batch_seq_minibatches(
                night_x,
                night_y,
                batch_size=config["batch_size"],
                seq_length=config["seq_length"],
            )
            # Evaluate
            test_outs = model.evaluate(test_minibatch_fn)
            s_trues.extend(test_outs["test/trues"])
            s_preds.extend(test_outs["test/preds"])
            trues.extend(test_outs["test/trues"])
            preds.extend(test_outs["test/preds"])

            # Save labels and predictions (each night of each subject)
            save_dict = {
                "y_true": test_outs["test/trues"],
                "y_pred": test_outs["test/preds"],
            }
            fname = os.path.basename(test_files[night_idx]).split(".")[0]
            save_path = os.path.join(
                output_dir,
                "pred_{}.npz".format(fname)
            )
            np.savez(save_path, **save_dict)
            logger.info("Saved outputs to {}".format(save_path))
    else:
        for night_idx, night_data in enumerate(zip(test_x, test_y)):
            # Create minibatches for testing
            night_x, night_y = night_data
            night_y = np.array([1])  # Dummy to make it work
            test_minibatch_fn = iterate_batch_multiple_seq_minibatches(
                [night_x],
                [night_y],
                batch_size=config["batch_size"],
                seq_length=config["seq_length"],
                shuffle_idx=None,
                augment_seq=False,
            )
            if (config.get('augment_signal') is not None) and config['augment_signal']:
                # Evaluate
                test_outs = model.evaluate_NEW(test_minibatch_fn)
            else:
                # Evaluate
                test_outs = model.evaluate(test_minibatch_fn)

            # Show result
            outcome = test_outs['test/preds']
            

    return outcome


# %% Test section

if __name__ == "__main__":
    model = importModel("./out_QS/train/17_new_v2")
    out = Predict_array(output_dir="./DataiBand/output/Fp1-Fp2_filtered",
                        args_log_file="info_ch_extract.log",
                        filtering_status=True,
                        lowcut=0.3,
                        highcut=30,
                        fs=256,
                        signal_req=np.random.rand(1, 7680),
                        signal_ref=np.random.rand(1, 7680),
                        model=model
                        )