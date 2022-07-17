'''
Authors: Anthony Badea, Jonathan Bossio
Date: Tuesday June 7, 2022
Purpose: Bootstrap uncertainties
'''

# python
import json
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
from plotting_functions import plot_loss

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
    
    # initialize bootstrap path
    bootstrap_path = ops.bootstrap_path
    if not os.path.isdir(bootstrap_path):
        os.makedirs(bootstrap_path)
    
    # write options to file to save
    with open(os.path.join(bootstrap_path,"options.json"),"w") as f:
        json.dump(vars(ops), f)

    # set seeds to get reproducible results (only if requested)
    if ops.seed is not None:
        try:
            python_random.seed(ops.seed)
            np.random.seed(opsseed)
            tf.random.set_seed(ops.seed)
        except:  # deprecated in newer tf versions
            tf.keras.utils.set_random_seed(ops.seed)

    #%%%%%%%%%%% Data loading %%%%%%%%%%%#
    # load this once
    with h5py.File(ops.inFile,"r") as f:
        # event selection
        minAvg = np.array(f['EventVars/minAvgMass_jetdiff10_btagdiff10'])
        cut_minAvg = 1000 # GeV
        mask = (minAvg > cut_minAvg)
        HT = np.array(f['EventVars/HT'])
        cut_HT = 1100 # GeV
        mask = np.logical_and(mask, HT > cut_HT)
        
        # load other variables
        HT = HT[mask]
        minAvg = minAvg[mask]
        dEta12 = np.array(f['EventVars/deta12'][mask])
        n_jets = np.array(f['EventVars/nJet'][mask])
        djmass = np.array(f['EventVars/djmass'][mask])
        normweight = np.array(f['EventVars/normweight'][mask])
        
        # concatenate
        x = np.stack([HT,minAvg,djmass],-1) # dEta12,n_jets,
        print(f"Number of events: {x.shape[0]}")

    # control and validation regions
    cut_deta12 = 1.5
    CR_njets, VR_njets, SR_njets = 5, 6, 7
    CR_high = np.logical_and(dEta12 >= cut_deta12, n_jets == CR_njets)
    CR_low  = np.logical_and(dEta12 < cut_deta12,  n_jets == CR_njets)
    VR_high = np.logical_and(dEta12 >= cut_deta12, n_jets == VR_njets)
    VR_low  = np.logical_and(dEta12 < cut_deta12,  n_jets == VR_njets)
    SR_high = np.logical_and(dEta12 >= cut_deta12, n_jets >= SR_njets)
    SR_low  = np.logical_and(dEta12 < cut_deta12,  n_jets >= SR_njets)

    # get events per region
    CR_high_x, CR_high_w = x[CR_high], normweight[CR_high]
    CR_low_x,   CR_low_w = x[CR_low],  normweight[CR_low]
    VR_high_x, VR_high_w = x[VR_high], normweight[VR_high]
    VR_low_x,   VR_low_w = x[VR_low],  normweight[VR_low]
    SR_high_x, SR_high_w = x[SR_high], normweight[SR_high]
    SR_low_x,   SR_low_w = x[SR_low],  normweight[SR_low]

    print(f"Events in CR_high {CR_high_x.shape}, CR_low {CR_high_x.shape}")
    print(f"Events in VR_high {VR_high_x.shape}, VR_low {VR_high_x.shape}")
    print(f"Events in SR_high {SR_high_x.shape}, SR_low {SR_high_x.shape}")

    # remove unused vars and cleanup
    del x, dEta12, n_jets, normweight
    gc.collect()

    # define training labels to reweight CR_high -> CR_low, i.e learning CR_low/CR_high
    CR_high_y = np.zeros(CR_high_x.shape[0])
    CR_low_y  = np.ones(CR_low_x.shape[0])

    #%%%%%%%%%%% Prepare Configurations %%%%%%%%%%%#
    confs = []
    for iB in range(ops.num_bootstraps):
        confs.append({
            "iB":iB,
            "bootstrap_path":bootstrap_path,
            "CR_high_x" : CR_high_x, "CR_high_w" : CR_high_w, "CR_high_y" : CR_high_y,
            "CR_low_x"  : CR_low_x,  "CR_low_w"  : CR_low_w,  "CR_low_y"  : CR_low_y,
            "VR_high_x" : VR_high_x, "VR_high_w" : VR_high_w,
            "VR_low_x"  : VR_low_x,  "VR_low_w"  : VR_low_w,
            "SR_high_x" : SR_high_x, "SR_high_w" : SR_high_w,
            "SR_low_x"  : SR_low_x,  "SR_low_w"  : SR_low_w,
        })

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

    # check out file name is ok
    outFileName = os.path.join(checkpoint_dir,"predictions.h5")
    if os.path.isfile(outFileName) and not ops.doOverwrite:
        logfile.write(f"Skipping because outfile name already exists: {outFileName}")
        return

    # concatenate
    X = np.concatenate([conf["CR_high_x"],conf["CR_low_x"]])
    Y = np.concatenate([conf["CR_high_y"],conf["CR_low_y"]])
    W = np.concatenate([conf["CR_high_w"],conf["CR_low_w"]])
    Y = np.stack([Y,W],axis=-1)

    # split data
    X_train, X_test, Y_train, Y_test, idx_train, idx_test = train_test_split(X, Y, np.arange(X.shape[0]), test_size=0.25, shuffle=True)
    del X, Y, W
    gc.collect()
    
    # write to log
    logfile.write(f"Train shapes ({X_train.shape},{Y_train.shape}), Test shapes ({X_test.shape},{Y_test.shape})" +"\n" )
    logfile.write(f"Train ones ({Y_train[:,0].sum()/Y_train.shape[0]}), Test ones ({Y_test[:,0].sum()/Y_test.shape[0]})" +"\n")

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
    
    # plot loss
    plot_loss(history, checkpoint_dir)

    # predict in each region
    CR_high_p = model.predict( conf["CR_high_x"], batch_size=10000).flatten()
    VR_high_p = model.predict( conf["VR_high_x"], batch_size=10000).flatten()
    SR_high_p = model.predict( conf["SR_high_x"], batch_size=10000).flatten()
    
    logfile.write(f"Saving to {outFileName}")
    # np.savez(outFileName, **outData)
    
    with h5py.File(outFileName, 'w') as hf:
        hf.create_dataset("idx_train", data = idx_train)
        hf.create_dataset("idx_test", data = idx_test)
        hf.create_dataset("CR_high_p", data = CR_high_p)
        hf.create_dataset("VR_high_p", data = VR_high_p)
        hf.create_dataset("SR_high_p", data = SR_high_p)

    # close
    logfile.write("Done closing log file")
    logfile.close()

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inFile", help="Input file.", default=None)
    parser.add_argument("-o", "--outDir", help="Output directory for plots", default="./")
    parser.add_argument("-nb", "--num_bootstraps", help="Number of trainings to perform for bootstrap.", default=1, type=int)
    parser.add_argument("-bp", "--bootstrap_path", help="Path where bootstrap saved to", default=os.path.join('./checkpoints', f'bootstrap_{datetime.datetime.now().strftime("%Y.%m.%d.%H.%M.%S")}'))
    parser.add_argument("-e", "--nepochs", help="Number of epochs.", default=1, type=int)
    parser.add_argument("-b", "--batch_size", help="Training batch size.", default=2048, type=int)
    parser.add_argument("-lr", "--learning_rate", help="Learning rate", default=1e-3, type=float)
    parser.add_argument("-s", "--seed", help="Seed for TensorFlow and NumPy", default=None, type=int)
    parser.add_argument("-j",  "--ncpu", help="Number of cores to use for multiprocessing. If not provided multiprocessing not done.", default=1, type=int)
    parser.add_argument("--doOverwrite", action="store_true", help="Overwrite existing output files with overlapping names.")
    return parser.parse_args()

if __name__ == "__main__":
    main()
