'''
Authors: Anthony Badea, Jonathan Bossio
Date: Monday April 25, 2022
'''

# python imports
import h5py
import tensorflow as tf
import tensorflow.keras.backend as K
import argparse
import json
import datetime
import os
import sys
import logging
import pandas as pd
import random as python_random
import numpy as np
from sklearn.model_selection import train_test_split
import gc

# custom imports
try:
    from make_model import simple_model
    from get_sample import get_sample
except:
    from ML_Keras.make_model import simple_model
    from ML_Keras.get_sample import get_sample

# Tensorflow GPU settings
physical_devices = tf.config.list_physical_devices('GPU') 
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# This function keeps the initial learning rate for the first N epochs and decreases it exponentially after that.
def scheduler(epoch, lr):
  if epoch < 5:
    return lr
  else:
    return lr * tf.math.exp(-0.1)

def main(config = None):

    # logger
    logging.basicConfig(level = 'INFO', format = '%(levelname)s: %(message)s')
    log = logging.getLogger()

    # user options
    ops = options()
    if config is not None:
        conf = config
    else:
        if ops.conf:
            with open(ops.conf) as f:
                log.info(f"opening {ops.conf}")
                conf = json.load(f)
        else:
            conf = {
                "file": ops.inFile,
                "nepochs": ops.nepochs,
                "batch_size": ops.batch_size,
                "learning_rate" : ops.learning_rate,
                "seed" : ops.seed
            }
            with open('conf.json', 'w') as fp:
                json.dump(conf, fp)

    # protection
    if not conf["file"]:
        log.error('ERROR: No input file was provided, exiting')
        sys.exit(1)

    # training configuration
    log.info("Training configuration: \n" + json.dumps(conf, indent=4, sort_keys=True))

    # data set generators
    seed = None
    if "seed" in conf and conf["seed"] is not None:
        seed = conf["seed"]
    
    # cuts used
    cut_minAvgMass = 750
    # grep ScoreCut /cvmfs/atlas.cern.ch/repo/sw/software/21.2/AnalysisBase/21.2.214/InstallArea/x86_64-centos7-gcc8-opt/data/BoostedJetTaggers/JetQGTaggerBDT/JetQGTaggerBDT*
    # 50%: (x<200)*(-0.000714*x-0.0121) + (x>=200)*-0.155, 80%: 0.05, 90%: 0.14
    cut_QGTaggerBDT = 0.14
    cut_nQuarkJets = 2
    
    # load this once
    with h5py.File(conf["file"],"r") as f:
        # pick up variables from file
        x = np.stack([
                np.array(f['EventVars']['HT']),
                np.array(f['EventVars']['deta']),
                np.array(f['EventVars']['djmass']),
                np.array(f['EventVars']['minAvgMass']),
                np.array(f['source']['pt'][:,0])
           ],-1)
        minAvgMass = np.array(f['EventVars']['minAvgMass'])
        nQuarkJets = (np.array(f['source']['QGTaggerBDT']) > cut_QGTaggerBDT).sum(1)
        normweight = np.array(f['normweight']['normweight'])
        print(f"Number of events: {minAvgMass.shape[0]}")

    # Create cuts to Reweight A -> C
    RegA = np.logical_and(minAvgMass < cut_minAvgMass, nQuarkJets < cut_nQuarkJets)
    RegC = np.logical_and(minAvgMass < cut_minAvgMass, nQuarkJets >= cut_nQuarkJets)
    print(f"Number of events in A and C: {RegA.sum()}, {RegC.sum()}")
    del minAvgMass, nQuarkJets
    gc.collect()

    # get events per region
    RegA_x = x[RegA]
    RegA_weights = normweight[RegA]
    RegA_y = np.zeros(RegA_weights.shape)
    RegC_x = x[RegC]
    RegC_weights = normweight[RegC]
    RegC_y = np.ones(RegC_weights.shape)
    del x, normweight
    gc.collect()
    
    # # normalize reg A and reg C to have same total weight to begin with
    # if RegC_weights.sum() > RegA_weights.sum():
    #     RegA_weights = RegC_weights.sum()/RegA_weights.sum() * RegA_weights
    # else:
    #     RegC_weights = RegA_weights.sum()/RegC_weights.sum() * RegC_weights
    # print(f"RegA/C_weights sum: {RegA_weights.sum()}, {RegC_weights.sum()}")

    # # shuffle
    # RegA_idx = np.random.choice(RegA_y.shape[0],RegA_y.shape[0])
    # RegA_ht = RegA_ht[RegA_idx]
    # RegA_weights = RegA_weights[RegA_idx]
    # RegA_y = RegA_y[RegA_idx]
    # RegC_idx = np.random.choice(RegC_y.shape[0],RegC_y.shape[0])
    # RegC_ht = RegC_ht[RegC_idx]
    # RegC_weights = RegC_weights[RegC_idx]
    # RegC_y = RegC_y[RegC_idx]
    
    # combine with same statistics
    nEventsA = -1 #min(RegA_y.shape[0],RegC_y.shape[0])
    nEventsC = -1 #2*nEvents
    X = np.concatenate([RegA_x[:nEventsA],RegC_x[:nEventsC]])
    Y = np.concatenate([RegA_y[:nEventsA],RegC_y[:nEventsC]])
    W = np.concatenate([RegA_weights[:nEventsA],RegC_weights[:nEventsC]])
    Y = np.stack([Y,W],axis=-1)
    del RegA_x, RegA_weights, RegA_y, RegC_x, RegC_weights, RegC_y
    gc.collect()

    # standardize
    X = (X - np.mean(X,0))/np.std(X,0)
    print(f"X mean, std: {np.mean(X)}, {np.std(X)}")
    
    # split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.75, shuffle=True)
    print(f"Train shapes ({X_train.shape},{Y_train.shape}), Test shapes ({X_test.shape},{Y_test.shape})")
    print(f"Train ones ({Y_train[:,0].sum()/Y_train.shape[0]}), Test ones ({Y_test[:,0].sum()/Y_test.shape[0]})")

    # set seeds to get reproducible results (only if requested)
    if seed is not None:
        try:
            python_random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)
        except:  # deprecated in newer tf versions
            tf.keras.utils.set_random_seed(seed)

    # make model
    model = simple_model(input_dim=X.shape[1], learning_rate=conf["learning_rate"])
    model.summary()

    # make callbacks
    callbacks = []
    callbacks.append(tf.keras.callbacks.EarlyStopping(patience=10, mode="min", restore_best_weights=True)) #, monitor="val_loss"))
    # ModelCheckpoint
    checkpoint_filepath = f'./checkpoints/training_{datetime.datetime.now().strftime("%Y.%m.%d.%H.%M.%S")}/' + "cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_filepath)
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(checkpoint_filepath, monitor="val_loss", mode="min", save_best_only=False, save_weights_only=True,))
    # Terminate on NaN such that it is easier to debug
    callbacks.append(tf.keras.callbacks.TerminateOnNaN())
    callbacks.append(tf.keras.callbacks.LearningRateScheduler(scheduler))

    # train
    history = model.fit(
        X_train, Y_train,
        batch_size=ops.batch_size,
        epochs=conf["nepochs"],
        callbacks=callbacks,
        verbose=1,
        validation_data=(X_test,Y_test)
    )

    # Get training history
    data = pd.DataFrame(history.history)
    log.debug(data.head())

    # Plot loss vs epochs (if nepochs > 1)
    if conf["nepochs"] > 1:
        try:
            from plotting_functions import plot_loss
        except:
            from ML_Keras.plotting_functions import plot_loss
        data['epoch'] = history.epoch
        plot_loss(history, ops.outDir)

    return data  # will be used in CI tests


def options():
    parser = argparse.ArgumentParser()
    # input files
    parser.add_argument("-c", "--conf", help="Configuration file. If provided, all other settings are overruled.", default=None)
    parser.add_argument("-i", "--inFile", help="Input file.", default=None)
    parser.add_argument("-o", "--outDir", help="Output directory for plots", default="./")
    # train settings
    parser.add_argument("-e", "--nepochs", help="Number of epochs.", default=1, type=int)
    parser.add_argument("-b", "--batch_size", help="Training batch size.", default=2048, type=int)
    parser.add_argument("-lr", "--learning_rate", help="Learning rate", default=1e-3, type=float)
    # model settings
    parser.add_argument("-s", "--seed", help="Seed for TensorFlow and NumPy", default=None, type=int)
    parser.add_argument("-ns", "--num_samples", help="Number of events", default=None, type=int)
    return parser.parse_args()

if __name__ == "__main__":
    main()
