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
from make_model import simple_model_norm, sqrtR_loss, mean_pred
from plotting_functions import plot_loss

# Need the following to run on LXPLUS
import matplotlib
matplotlib.use('Agg')

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

    # cuts used
    cut_minAvgMass = 750
    cut_deta = 1.4
    # grep ScoreCut /cvmfs/atlas.cern.ch/repo/sw/software/21.2/AnalysisBase/21.2.214/InstallArea/x86_64-centos7-gcc8-opt/data/BoostedJetTaggers/JetQGTaggerBDT/JetQGTaggerBDT*
    # 50%: (x<200)*(-0.000714*x-0.0121) + (x>=200)*-0.155, 80%: 0.05, 90%: 0.14
    cut_QGTaggerBDT = 0.14
    cut_nQuarkJets = 2
    
    # load this once
    with h5py.File(ops.inFile,"r") as f:
        # pick up variables from file
        tostack = [
                np.array(f['EventVars']['HT']),
                np.array(f['EventVars']['deta']),
                np.array(f['EventVars']['djmass']),
                np.array(f['source']['pt'][:,0])
           ]
        if not ops.no_minavg:
            tostack.append( np.array(f['EventVars']['minAvgMass']) )
        x = np.stack(tostack, -1)
        if ops.more_vars:
            x2 = np.stack([
                    np.array(f['source']['pt'][:,1]),
                    np.array(f['source']['pt'][:,2]),
                    np.array(f['source']['pt'][:,3]),
                    np.array(f['source']['pt'][:,4]),
                    np.array(f['source']['pt'][:,5]),

               ],-1)
            x = np.concatenate([x,x2],axis=-1)

        deta = np.array(f['EventVars']['deta'])
        minAvgMass = np.array(f['EventVars']['minAvgMass'])
        nQuarkJets = (np.array(f['source']['QGTaggerBDT']) > cut_QGTaggerBDT).sum(1)
        normweight = np.array(f['normweight']['normweight'])
        # normweight = np.sqrt(normweight)
        print(f"Number of events: {minAvgMass.shape[0]}")

    # Create cuts to Reweight A -> C
    if ops.SR2D:
        SRcut = np.logical_and(minAvgMass >= cut_minAvgMass, deta < cut_deta)
    else:
        SRcut = minAvgMass >= cut_minAvgMass
    Reg0qincl = nQuarkJets == 0
    Reg1qincl = nQuarkJets == 1
    Reg1qCR = np.logical_and(nQuarkJets == 1, np.logical_not(SRcut))
    Reg2qCR = np.logical_and(nQuarkJets >= 2, np.logical_not(SRcut))
    Reg1qSR = np.logical_and(nQuarkJets == 1, SRcut)
    Reg2qSR = np.logical_and(nQuarkJets >= 2, SRcut)

    print(f"Number of events in training regions: {Reg0qincl.sum(), Reg1qincl.sum(), Reg1qCR.sum(), Reg2qCR.sum()}")
    print(f"Number of events in analysis regions: {Reg1qSR.sum(), Reg2qSR.sum()}")
    del minAvgMass, nQuarkJets, deta
    gc.collect()

    # get events per region
    Reg0qincl_x = x[Reg0qincl]
    np.append(Reg0qincl_x,  np.zeros((Reg0qincl_x.shape[0],1) )) #add theta=0 to the x vars
    Reg0qincl_weights = normweight[Reg0qincl]
    Reg0qincl_y = np.zeros(Reg0qincl_weights.shape)
    Reg1qincl_x = x[Reg1qincl]
    np.append(Reg1qincl_x,  np.zeros((Reg1qincl_x.shape[0],1) )) #add theta=0 to the x vars
    Reg1qincl_weights = normweight[Reg1qincl]
    Reg1qincl_y = np.ones(Reg1qincl_weights.shape)
    Reg1qCR_x = x[Reg1qCR]
    np.append(Reg1qCR_x,  np.ones((Reg1qCR_x.shape[0],1) )) #add theta=1 to the x vars
    Reg1qCR_weights = normweight[Reg1qCR]
    Reg1qCR_y = np.zeros(Reg1qCR_weights.shape)
    Reg2qCR_x = x[Reg2qCR]
    np.append(Reg2qCR_x,  np.ones((Reg2qCR_x.shape[0],1) )) #add theta=1 to the x vars
    Reg2qCR_weights = normweight[Reg2qCR]
    Reg2qCR_y = np.ones(Reg2qCR_weights.shape)

    # just for evaluation get region B and D
    Reg1qSR_x = x[Reg1qSR]
    np.append(Reg1qSR_x,  np.ones((Reg1qSR_x.shape[0],1) )) #add theta=1 to the x vars
    Reg2qSR_x = x[Reg2qSR]
    np.append(Reg2qSR_x,  np.ones((Reg2qSR_x.shape[0],1) )) #add theta=1 to the x vars
    del x, normweight
    gc.collect()

    # prepare confs
    confs = []
    for iB in range(ops.num_bootstraps):
        confs.append({
            "iB":iB,
            "bootstrap_path":bootstrap_path,
            "Reg0qincl_x":Reg0qincl_x, "Reg0qincl_y":Reg0qincl_y, "Reg0qincl_weights":Reg0qincl_weights,
            "Reg1qincl_x":Reg1qincl_x, "Reg1qincl_y":Reg1qincl_y, "Reg1qincl_weights":Reg1qincl_weights,
            "Reg1qCR_x":Reg1qCR_x, "Reg1qCR_y":Reg1qCR_y, "Reg1qCR_weights":Reg1qCR_weights,
            "Reg2qCR_x":Reg2qCR_x, "Reg2qCR_y":Reg2qCR_y, "Reg2qCR_weights":Reg2qCR_weights,
            "Reg1qSR_x":Reg1qSR_x, 
            "Reg2qSR_x":Reg2qSR_x
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

    # concatenate
    X = np.concatenate([conf["Reg0qincl_x"],conf["Reg1qincl_x"],conf["Reg1qCR_x"],conf["Reg2qCR_x"]])
    Y = np.concatenate([conf["Reg0qincl_y"],conf["Reg1qincl_y"],conf["Reg1qCR_y"],conf["Reg2qCR_y"]])
    W = np.concatenate([conf["Reg0qincl_weights"],conf["Reg1qincl_weights"],conf["Reg1qCR_weights"],conf["Reg2qCR_weights"]])
    Y = np.stack([Y,W],axis=-1)

    # standardize
    #X = (X - np.mean(X,0))/np.std(X,0)
    #logfile.write(f"X mean, std: {np.mean(X)}, {np.std(X)}")

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
    model = simple_model_norm(input_dim=X_train.shape[1])
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=ops.learning_rate), loss=sqrtR_loss, metrics=[mean_pred])

    # fit
    history = model.fit(
        X_train, Y_train,
        batch_size=ops.batch_size,
        epochs=ops.nepochs,
        callbacks=callbacks,
        verbose=1,
        shuffle=True,
        class_weight={0:1., 1:0.25},
        validation_data=(X_test,Y_test)
    )
    
    # plot loss
    plot_loss(history, checkpoint_dir)

    # predict in each region
    logfile.write(f"Prediction Region 0qincl with {conf['Reg0qincl_x'].shape[0]} events" + "\n")
    Reg0qincl_p = model.predict(conf["Reg0qincl_x"],batch_size=10000).flatten()
    logfile.write(f"Prediction Region 1qincl with {conf['Reg1qincl_x'].shape[0]} events" + "\n")
    Reg1qincl_p = model.predict(conf["Reg1qincl_x"],batch_size=10000).flatten()
    logfile.write(f"Prediction Region 1qCR with {conf['Reg1qCR_x'].shape[0]} events" + "\n")
    Reg1qCR_p = model.predict(conf["Reg1qCR_x"],batch_size=10000).flatten()
    logfile.write(f"Prediction Region 2qCR with {conf['Reg2qCR_x'].shape[0]} events" + "\n")
    Reg2qCR_p = model.predict(conf["Reg2qCR_x"],batch_size=10000).flatten()
    logfile.write(f"Prediction Region 1qSR with {conf['Reg1qSR_x'].shape[0]} events" + "\n")
    Reg1qSR_p = model.predict(conf["Reg1qSR_x"],batch_size=10000).flatten()
    logfile.write(f"Prediction Region 2qSR with {conf['Reg2qSR_x'].shape[0]} events" + "\n")
    Reg2qSR_p = model.predict(conf["Reg2qSR_x"],batch_size=10000).flatten()
    np.savez(
        os.path.join(checkpoint_dir,"predictions.npz"),
        **{
        "idx_train": idx_train, "idx_test": idx_test,
        "Reg0qincl_p": Reg0qincl_p,
        "Reg1qincl_p": Reg1qincl_p,
        "Reg1qCR_p": Reg1qCR_p,
        "Reg2qCR_p": Reg2qCR_p,
        "Reg1qSR_p": Reg1qSR_p,
        "Reg2qSR_p": Reg2qSR_p})

    # close
    logfile.write("Done closing log file")
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
    parser.add_argument("--SR2D", action="store_true", help="Define 2D SR")
    parser.add_argument("--more-vars", action="store_true", help="Include also all jet pts")
    parser.add_argument("--no-minavg", action="store_true", help="Skip minavg from inputs")
    return parser.parse_args()

if __name__ == "__main__":
    main()
