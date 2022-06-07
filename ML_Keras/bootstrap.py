'''
Authors: Anthony Badea, Jonathan Bossio
Date: Tuesday June 7, 2022
Purpose: Bootstrap uncertainties
'''

# python
import h5py
import argparse
import numpy as np
import gc
import tensorflow as tf
from sklearn.model_selection import train_test_split
import datetime
import os
import multiprocessing as mp

# custom
from make_model import simple_model, sqrtR_loss, mean_pred

# Tensorflow GPU settings
physical_devices = tf.config.list_physical_devices('GPU') 
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# multiprocessing settings
mp.set_start_method('spawn', force=True)

# This function keeps the initial learning rate for the first N epochs and decreases it exponentially after that.
def scheduler(epoch, lr):
  if epoch < 5:
    return lr
  else:
    return lr * tf.math.exp(-0.1)

def main(config = None):

    # user options
    ops = options()

    # set seeds to get reproducible results (only if requested)
    if ops.seed is not None:
        try:
            python_random.seed(ops.seed)
            np.random.seed(opsseed)
            tf.random.set_seed(ops.seed)
        except:  # deprecated in newer tf versions
            tf.keras.utils.set_random_seed(ops.seed)

    # cuts used
    cut_minAvgMass = 750
    # grep ScoreCut /cvmfs/atlas.cern.ch/repo/sw/software/21.2/AnalysisBase/21.2.214/InstallArea/x86_64-centos7-gcc8-opt/data/BoostedJetTaggers/JetQGTaggerBDT/JetQGTaggerBDT*
    # 50%: (x<200)*(-0.000714*x-0.0121) + (x>=200)*-0.155, 80%: 0.05, 90%: 0.14
    cut_QGTaggerBDT = 0.14
    cut_nQuarkJets = 2
    
    # load this once
    with h5py.File(ops.inFile,"r") as f:
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
        # normweight = np.sqrt(normweight)
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

    # concatenate
    X = np.concatenate([RegA_x,RegC_x])
    Y = np.concatenate([RegA_y,RegC_y])
    W = np.concatenate([RegA_weights,RegC_weights])
    Y = np.stack([Y,W],axis=-1)
    del RegA_x, RegA_weights, RegA_y, RegC_x, RegC_weights, RegC_y
    gc.collect()

    # standardize
    X = (X - np.mean(X,0))/np.std(X,0)
    print(f"X mean, std: {np.mean(X)}, {np.std(X)}")

    # prepare confs
    confs = []
    for iB in range(ops.num_bootstraps):
        confs.append({"iB":iB,"X":X,"Y":Y,"bootstrap_path":ops.bootstrap_path})

    # launch jobs
    if ops.ncpu == 1:
        for conf in confs:
            train(conf)
    else:
        results = mp.Pool(ops.ncpu).map(train, confs)

def train(conf):

    # user options
    ops = options()

    # ModelCheckpoint
    checkpoint_filepath = os.path.join(conf["bootstrap_path"], f'training_{conf["iB"]}', "cp-{epoch:04d}.ckpt")
    checkpoint_dir = os.path.dirname(checkpoint_filepath)
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # make log file
    logfile = open(os.path.join(checkpoint_dir,"log.txt"),"w")

    # split data
    X_train, X_test, Y_train, Y_test = train_test_split(conf["X"], conf["Y"], test_size=0.75, shuffle=True)
    logfile.write(f"Train shapes ({X_train.shape},{Y_train.shape}), Test shapes ({X_test.shape},{Y_test.shape})" +"\n" )
    logfile.write(f"Train ones ({Y_train[:,0].sum()/Y_train.shape[0]}), Test ones ({Y_test[:,0].sum()/Y_test.shape[0]})" +"\n")
    del conf
    gc.collect()

    # make callbacks
    callbacks = []
    callbacks.append(tf.keras.callbacks.EarlyStopping(patience=30, mode="min", restore_best_weights=True)) #, monitor="val_loss"))
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(checkpoint_filepath, monitor="val_loss", mode="min", save_best_only=False, save_weights_only=True,))
    # Terminate on NaN such that it is easier to debug
    callbacks.append(tf.keras.callbacks.TerminateOnNaN())
    callbacks.append(tf.keras.callbacks.LearningRateScheduler(scheduler))
    callbacks.append(tf.keras.callbacks.CSVLogger(os.path.join(checkpoint_dir,"fit.txt"), separator=",", append=False))

    # compile
    model = simple_model(input_dim=X_train.shape[1])
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=ops.learning_rate), loss=sqrtR_loss, metrics=[mean_pred])

    # fit
    history = model.fit(
        X_train, Y_train,
        batch_size=ops.batch_size,
        epochs=ops.nepochs,
        callbacks=callbacks,
        verbose=1,
        validation_data=(X_test,Y_test)
    )

    # close
    logfile.close()

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inFile", help="Input file.", default=None)
    parser.add_argument("-o", "--outDir", help="Output directory for plots", default="./")
    parser.add_argument("-nb", "--num_bootstraps", help="Number of trainings to perform for bootstrap.", default=2, type=int)
    parser.add_argument("-bp", "--bootstrap_path", help="Path where bootstrap saved to", default=os.path.join('./checkpoints', f'bootstrap_{datetime.datetime.now().strftime("%Y.%m.%d.%H.%M.%S")}'))
    parser.add_argument("-e", "--nepochs", help="Number of epochs.", default=1, type=int)
    parser.add_argument("-b", "--batch_size", help="Training batch size.", default=2048, type=int)
    parser.add_argument("-lr", "--learning_rate", help="Learning rate", default=1e-3, type=float)
    parser.add_argument("-s", "--seed", help="Seed for TensorFlow and NumPy", default=None, type=int)
    parser.add_argument("-j",  "--ncpu", help="Number of cores to use for multiprocessing. If not provided multiprocessing not done.", default=1, type=int)
    return parser.parse_args()

if __name__ == "__main__":
    main()
