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

# custom imports
try:
    from make_model import make_model
    # from get_data import get_data
    from get_sample import get_sample
except:
    from ML_Keras.make_model import make_model
    # from ML_Keras.get_data import get_data
    from ML_Keras.get_sample import get_sample

# Tensorflow GPU settings
physical_devices = tf.config.list_physical_devices('GPU') 
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def weighted_binary_crossentropy(y_true, y_pred):
    weights = tf.gather(y_true, [1], axis=1) # event weights
    y_true = tf.gather(y_true, [0], axis=1) # actual y_true for loss

    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    t_loss = -1 * ((y_true) * K.log(y_pred + epsilon) + (1 - y_true) * K.log(1 - y_pred + epsilon)) * weights 
    t_loss = K.mean(t_loss)
    # tf.print(y_true)
    # tf.print(t_loss)
    #tf.print(tf.reduce_sum(tf.cast((y_true - y_pred)<=0.1, tf.float32)))
    return t_loss

def my_accuracy(y_true, y_pred):
    y_true = tf.gather(y_true, [0], axis=1) # actual y_true for loss
    return tf.reduce_sum(tf.cast((y_true - y_pred)<=0.1, tf.float32))

def main(config = None):

    # logger
    logging.basicConfig(level = 'INFO', format = '%(levelname)s: %(message)s')
    log = logging.getLogger()

    # user options
    if config is not None:
        conf = config
    else:
        ops = options()
        if ops.conf:
            with open(ops.conf) as f:
                log.info(f"opening {ops.conf}")
                conf = json.load(f)
        else:
            conf = {
                "file": ops.inFile,
                "nepochs": ops.nepochs,
                "train_batch_size": ops.train_batch_size,
                "val_batch_size": ops.val_batch_size,
                "validation_steps" : ops.validation_steps,
                "learning_rate" : ops.learning_rate,
                "input_dim" : ops.input_dim,
                "ndense" : ops.ndense,
                "nnode_per_dense" : ops.nnode_per_dense,
                "seed" : ops.seed,
                "num_samples" : ops.num_samples
            }
            with open('conf.json', 'w') as fp:
                json.dump(conf, fp)

    # protection
    if not conf["file"]:
        log.error('ERROR: No input file was provided, exiting')
        sys.exit(1)

    # training configuration
    if "num_samples" not in conf or conf["num_samples"] is None:  # use size of input sample
        with h5py.File(conf["file"],"r") as hf:
            # conf["num_samples"] = hf["nQuarkJets"]['values'].shape[0]
            conf["num_samples"] = hf['EventVars']['HT'].shape[0]
    conf["train_steps_per_epoch"] = conf["num_samples"] // conf["train_batch_size"]
    log.info("Training configuration: \n" + json.dumps(conf, indent=4, sort_keys=True))

    # data set generators
    seed = None
    if "seed" in conf and conf["seed"] is not None:
        seed = conf["seed"]
    # train_data_gen = get_data(conf["file"], conf["nepochs"], conf["train_batch_size"], seed)
    # # sample validation data from the same probability density function (but generated val data is statistically independent w.r.t training data)
    # val_data_gen = get_data(conf["file"], conf["nepochs"], conf["val_batch_size"], seed+1 if seed is not None else None)

    # load this once
    
    f = h5py.File(conf["file"],"r")
    # cuts used
    cut_minAvgMass = 750
    # grep ScoreCut /cvmfs/atlas.cern.ch/repo/sw/software/21.2/AnalysisBase/21.2.214/InstallArea/x86_64-centos7-gcc8-opt/data/BoostedJetTaggers/JetQGTaggerBDT/JetQGTaggerBDT*
    # 50%: (x<200)*(-0.000714*x-0.0121) + (x>=200)*-0.155
    # 80%: 0.05
    # 90%: 0.14
    cut_QGTaggerBDT = 0.14
    cut_nQuarkJets = 2
    # precompute indices
    # minAvgMass = np.array(f['EventVars']['minAvgMass'])
    # low_minAvgMass = np.where(minAvgMass < cut_minAvgMass)[0] 
    # # only keep these loaded in memory
    # minAvgMass = minAvgMass[low_minAvgMass]
    # normweight = np.array(f['normweight']['normweight'])[low_minAvgMass]
    # #normweight = np.ones(normweight.shape)
    # nQuarkJets = (np.array(f['source']['QGTaggerBDT']) > cut_QGTaggerBDT).sum(1)[low_minAvgMass]
    # # fourmom = np.stack([f['source']['mass'], f['source']['pt'], f['source']['eta'], f['source']['phi']],-1)[low_minAvgMass]
    # HT = np.array(f['EventVars']['HT'])[low_minAvgMass] 
    # # # place ht cut
    # # low_HT = np.where(HT < 3000)[0]
    # # HT = HT[low_HT]
    # # nQuarkJets = nQuarkJets[low_HT]
    # # normweight = normweight[low_HT]
    # # standardize
    # HT = (HT - np.mean(HT))/np.std(HT)
    # Y = np.stack([nQuarkJets >= cut_nQuarkJets, normweight],axis=-1)
    # HT_train, HT_test, Y_train, Y_test = train_test_split(HT, Y, test_size=0.5, shuffle=True)
    # print(f"Train shapes ({HT_train.shape},{Y_train.shape}), Test shapes ({HT_test.shape},{Y_test.shape})")
    # print(f"Train ones ({Y_train[:,0].sum()/Y_train.shape[0]}), Test ones ({Y_test[:,0].sum()/Y_test.shape[0]})")
    
    # pick up variables from file
    #HT = np.array(f['EventVars']['HT'])
    HT = np.stack([np.array(f['EventVars']['HT']),
                   np.array(f['EventVars']['deta']),
                   np.array(f['EventVars']['djmass']),
                   np.array(f['EventVars']['minAvgMass']),
                   #np.array(f['source']['pt'][:,0])
               ],-1)
    minAvgMass = np.array(f['EventVars']['minAvgMass'])
    nQuarkJets = (np.array(f['source']['QGTaggerBDT']) > cut_QGTaggerBDT).sum(1)
    normweight = np.array(f['normweight']['normweight'])
    #normweight = np.ones(normweight.shape)
    print(f"Number of events: {minAvgMass.shape[0]}")
    # Create cuts to Reweight A -> C
    RegA = np.logical_and(minAvgMass < cut_minAvgMass, nQuarkJets < cut_nQuarkJets)
    RegC = np.logical_and(minAvgMass < cut_minAvgMass, nQuarkJets >= cut_nQuarkJets)
    print(f"Number of events in A and C: {RegA.sum()}, {RegC.sum()}")
    # get events per region
    RegA_ht = HT[RegA]
    RegA_weights = normweight[RegA]
    RegA_y = np.zeros(RegA_weights.shape) #np.zeros(RegA_weights.shape)
    RegC_ht = HT[RegC]
    RegC_weights = normweight[RegC]
    RegC_y = np.ones(RegC_weights.shape)
    # normalize reg A and reg C to have same total weight to begin with
    if RegC_weights.sum() > RegA_weights.sum():
        RegA_weights = RegC_weights.sum()/RegA_weights.sum() * RegA_weights
    else:
        RegC_weights = RegA_weights.sum()/RegC_weights.sum() * RegC_weights
    print(f"RegA/C_weights sum: {RegA_weights.sum()}, {RegC_weights.sum()}")
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
    nEvents = -1 #min(RegA_y.shape[0],RegC_y.shape[0])
    nEventsC = -1 #2*nEvents
    X = np.concatenate([RegA_ht[:nEvents],RegC_ht[:nEventsC]])
    Y = np.concatenate([RegA_y[:nEvents],RegC_y[:nEventsC]])
    W = np.concatenate([RegA_weights[:nEvents],RegC_weights[:nEventsC]])
    Y = np.stack([Y,W],axis=-1)
    
    # standardize
    X = (X - np.mean(X,0))/np.std(X,0)
    #X = (X - np.mean(X))/np.std(X)
    print(np.mean(X), np.std(X), X.shape, Y.shape)
    
    # split
    HT_train, HT_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.85, shuffle=True)
    print(f"Train shapes ({HT_train.shape},{Y_train.shape}), Test shapes ({HT_test.shape},{Y_test.shape})")
    print(f"Train ones ({Y_train[:,0].sum()/Y_train.shape[0]}), Test ones ({Y_test[:,0].sum()/Y_test.shape[0]})")

    # probabilities = normweight / normweight.sum()
    # probabilities = np.ones(normweight.shape)/normweight.shape[0] # uniform probability but weight the loss function
    # train_data_gen = get_sample(HT, nQuarkJets, normweight, probabilities, conf["train_batch_size"], cut_nQuarkJets) # update to four momentum when ready
    # val_data_gen = get_sample(HT, nQuarkJets, normweight, probabilities, conf["val_batch_size"], cut_nQuarkJets)
    
    # set seeds to get reproducible results (only if requested)
    if seed is not None:
        try:
            python_random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)
        except:  # deprecated in newer tf versions
            tf.keras.utils.set_random_seed(seed)

    # make model
    model = make_model(input_dim=HT.shape[1], ndense=conf["ndense"], nnode_per_dense=conf["nnode_per_dense"], learning_rate=conf["learning_rate"], loss=weighted_binary_crossentropy)
    model.summary()

    # make callbacks
    callbacks = []
    # EarlyStopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        patience=30, mode="min", restore_best_weights=True #, monitor="val_loss"
    )
    callbacks.append(early_stopping)
    # ModelCheckpoint
    checkpoint_filepath = f'./checkpoints/training_{datetime.datetime.now().strftime("%Y.%m.%d.%H.%M.%S")}/' + "cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_filepath)
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_filepath, monitor="myacc", mode="min", save_best_only=True, save_weights_only=True,
    )
    callbacks.append(model_checkpoint)
    # Terminate on NaN such that it is easier to debug
    callbacks.append(tf.keras.callbacks.TerminateOnNaN())

    # train
    # history = model.fit(
    #     train_data_gen,
    #     steps_per_epoch=conf["train_steps_per_epoch"],
    #     epochs=conf["nepochs"],
    #     callbacks=callbacks,
    #     verbose=1,
    #     validation_data=val_data_gen,
    #     validation_steps=conf["validation_steps"]
    # )

    history = model.fit(
        HT_train, Y_train,
        batch_size=ops.train_batch_size,
        epochs=conf["nepochs"],
        callbacks=callbacks,
        verbose=1,
        validation_data=(HT_test,Y_test)
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
        plot_loss(history)

    return data  # will be used in CI tests


def options():
    parser = argparse.ArgumentParser()
    # input files
    parser.add_argument("-c",  "--conf", help="Configuration file. If provided, all other settings are overruled.", default=None)
    parser.add_argument("-i",  "--inFile", help="Input file.", default=None)
    parser.add_argument("-o", "--outDir", help="Output directory for plots", default="./")
    # train settings
    parser.add_argument("-ne", "--nepochs", help="Number of epochs.", default=1, type=int)
    parser.add_argument("-tb", "--train_batch_size", help="Training batch size.", default=2048, type=int)
    parser.add_argument("-vb", "--val_batch_size", help="Validation batch size.", default=256, type=int)
    parser.add_argument("-vs", "--validation_steps", help="Number of validation steps.", default=1, type=int)
    parser.add_argument("-lr", "--learning_rate", help="Learning rate", default=1e-3, type=float)
    # model settings
    parser.add_argument("-ni", "--input_dim", help="Dimension of inputs per event for the first layer.", default=1, type=int)
    parser.add_argument("-nl", "--ndense", help="Number of dense layers.", default=1, type=int)
    parser.add_argument("-nd", "--nnode_per_dense", help="Number of nodes per dense layer.", default=30, type=int)
    parser.add_argument("-s", "--seed", help="Seed for TensorFlow and NumPy", default=None, type=int)
    parser.add_argument("-ns", "--num_samples", help="Number of events", default=None, type=int)
    return parser.parse_args()

if __name__ == "__main__":
    main()
