'''
Authors: Anthony Badea, Jonathan Bossio
Date: Monday April 25, 2022
'''

# Need the following to run on LXPLUS
import matplotlib
matplotlib.use('Agg')

# python imports
import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import tensorflow as tf
import os
import sys
import logging
import glob
import gc

# custom code
try:
    from get_data import get_full_data
    from make_model import simple_model
except:
    from ML_Keras.get_data import get_full_data
    from ML_Keras.make_model import simple_model

# Tensorflow GPU settings
physical_devices = tf.config.list_physical_devices('GPU') 
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# matplotlib
plt.rcParams['figure.figsize'] = (4,4)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True
colors = [
    "#E24A33", # orange
    "#7A68A6", # purple
    "#348ABD", # blue
    "#188487", # turquoise
    "#A60628", # red
    "#CF4457", # pink
    "#467821", # green
]

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
            }

    # protection
    if ops.model_weights is None:
        log.error('ERROR: no model weights were provided, exiting')
        sys.exit(1)

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
        # normweight = np.sqrt(normweight)
        print(f"Number of events: {minAvgMass.shape[0]}")

    # Create cuts to Reweight A -> C
    RegA = np.logical_and(minAvgMass < cut_minAvgMass, nQuarkJets < cut_nQuarkJets)
    RegC = np.logical_and(minAvgMass < cut_minAvgMass, nQuarkJets >= cut_nQuarkJets)
    RegB = np.logical_and(minAvgMass >= cut_minAvgMass, nQuarkJets < cut_nQuarkJets)
    RegD = np.logical_and(minAvgMass >= cut_minAvgMass, nQuarkJets >= cut_nQuarkJets)
    print(f"Number of events in A and C: {RegA.sum()}, {RegC.sum()}")
    # print(f"Number of events in B and D: {RegB.sum()}, {RegD.sum()}")
    del minAvgMass, nQuarkJets
    gc.collect()

    # get events per region
    RegA_x, RegA_weights = x[RegA], normweight[RegA]
    RegA_ht = RegA_x[:,0]
    RegC_x, RegC_weights = x[RegC], normweight[RegC]
    RegC_ht = RegC_x[:,0]
    RegB_x, RegB_weights = x[RegB], normweight[RegB]
    RegB_ht = RegB_x[:,0]
    RegD_x, RegD_weights = x[RegD], normweight[RegD]
    RegD_ht = RegD_x[:,0]
    del x, normweight
    gc.collect()

    # normalize for prediction
    RegA_x = (RegA_x-np.mean(RegA_x,0))/np.std(RegA_x,0)
    RegC_x = (RegC_x-np.mean(RegC_x,0))/np.std(RegC_x,0)
    RegB_x = (RegB_x-np.mean(RegB_x,0))/np.std(RegB_x,0)
    RegD_x = (RegD_x-np.mean(RegD_x,0))/np.std(RegD_x,0)

    # load model
    model = simple_model(input_dim=RegA_x.shape[1], learning_rate=1e-3)
    model.summary()
    # if checkpoint directory provided use the latest
    if os.path.isdir(ops.model_weights):
        latest = tf.train.latest_checkpoint(ops.model_weights)
        log.info(f"Using latest weights from checkpoint directory: {latest}")
        model.load_weights(latest).expect_partial()
    elif ops.model_weights == "1":
        latest = tf.train.latest_checkpoint(glob.glob("checkpoints/*")[-1])
        log.info(f"Using latest weights from checkpoint directory: {latest}")
        model.load_weights(latest).expect_partial()
    else:
        model.load_weights(ops.model_weights).expect_partial()

    # predict
    RegA_p = model.predict(RegA_x).flatten()
    print(f"RegA_p: {np.mean(RegA_p)},{np.std(RegA_p)}")
    RegA_reweighted = RegA_weights * np.exp(RegA_p) # #np.nan_to_num(RegA_p/(1-RegA_p),posinf=1)
    RegC_p = -999 #model.predict(RegC_x).flatten()

    # plot
    fig, [ax,rx] = plt.subplots(2,1,constrained_layout=False,sharey=False,sharex=True,gridspec_kw={"height_ratios": [3.5,1], 'hspace':0.0},)
    rx.set_ylabel("Ratio\nTo RegC")
    rx.set_xlabel(r"H$_{\mathrm{T}}$ [GeV]")
    rx.set_ylim(0,2)
    ax.set_ylabel("Density of Events")
    ax.set_yscale("log")
    bins = np.linspace(0, 13000, 100)
    c0, bin_edges, _ = ax.hist(RegA_ht, bins = bins, weights = RegA_weights, label = rf'RegA NQuarkJets $<$ {cut_nQuarkJets}', color = colors[0], density=False, histtype="step", lw=2)
    c1, bin_edges, _ = ax.hist(RegC_ht, bins = bins, weights = RegC_weights, label = rf'RegC NQuarkJets $\geq$ {cut_nQuarkJets}', color = colors[1], density=False, histtype="step", lw=2)
    c2, bin_edges, _ = ax.hist(RegA_ht, bins = bins, weights = RegA_reweighted, label = rf'Reweight RegA $\rightarrow$ RegC', color = colors[2], density=False, histtype="step", lw=2) 
    rx.plot((bin_edges[:-1] + bin_edges[1:]) / 2, c0/(c1 + 10**-50), 'o-', label = rf'RegA $/$ RegC', color = colors[0], lw=1)
    rx.plot((bin_edges[:-1] + bin_edges[1:]) / 2, c2/(c1 + 10**-50), 'o-', label = rf'Reweighted RegA $/$ RegC', color = colors[2], lw=1)
    rx.plot((bin_edges[:-1] + bin_edges[1:]) / 2,[1] * len((bin_edges[:-1] + bin_edges[1:]) / 2), ls="--",color="black",alpha=0.8)
    ax.legend(title=rf"minAvgMass $<$ {cut_minAvgMass} GeV", loc="upper right", prop={'size': 8}, framealpha=0.0)
    # rx.legend(title="", loc="best", prop={'size': 7}, framealpha=0.0)
    plt.savefig(os.path.join(ops.outDir,'reweightAtoC.pdf'), bbox_inches="tight")

    # Reweight B -> D
    RegB_p = model.predict(RegB_x).flatten()
    print(f"RegB_p: {np.mean(RegB_p)},{np.std(RegB_p)}")
    RegB_reweighted = RegB_weights * np.exp(RegB_p) # #np.nan_to_num(RegB_p/(1-RegB_p),posinf=1)
    RegD_p = -999 #model.predict(RegD_x).flatten()

    # plot
    fig, [ax,rx] = plt.subplots(2,1,constrained_layout=False,sharey=False,sharex=True,gridspec_kw={"height_ratios": [3.5,1], 'hspace':0.0},)
    rx.set_ylabel("Ratio\nTo RegD")
    rx.set_xlabel(r"H$_{\mathrm{T}}$ [GeV]")
    rx.set_ylim(0,2)
    ax.set_ylabel("Density of Events")
    ax.set_yscale("log")
    bins = np.linspace(0, 13000, 100)
    c0, bin_edges, _ = ax.hist(RegB_ht, bins = bins, weights = RegB_weights, label = rf'RegB NQuarkJets $<$ {cut_nQuarkJets}', color = colors[0], density=False, histtype="step", lw=2)
    c1, bin_edges, _ = ax.hist(RegD_ht, bins = bins, weights = RegD_weights, label = rf'RegD NQuarkJets $\geq$ {cut_nQuarkJets}', color = colors[1], density=False, histtype="step", lw=2)
    c2, bin_edges, _ = ax.hist(RegB_ht, bins = bins, weights = RegB_reweighted, label = rf'Reweight RegB $\rightarrow$ RegD', color = colors[2], density=False, histtype="step", lw=2) 
    rx.plot((bin_edges[:-1] + bin_edges[1:]) / 2, c0/(c1 + 10**-50), 'o-', label = rf'RegB $/$ RegD', color = colors[0], lw=1)
    rx.plot((bin_edges[:-1] + bin_edges[1:]) / 2, c2/(c1 + 10**-50), 'o-', label = rf'Reweighted RegB $/$ RegD', color = colors[2], lw=1)
    rx.plot((bin_edges[:-1] + bin_edges[1:]) / 2,[1] * len((bin_edges[:-1] + bin_edges[1:]) / 2), ls="--",color="black",alpha=0.8)
    ax.legend(title=rf"minAvgMass $\geq$ {cut_minAvgMass} GeV", loc="upper left", prop={'size': 8}, framealpha=0.0)
    # rx.legend(title="", loc="best", prop={'size': 7}, framealpha=0.0)
    plt.savefig(os.path.join(ops.outDir,'reweightBtoD.pdf'), bbox_inches="tight")
    
    np.savez("preds.npz",**{"RegA_ht":RegA_ht,"RegA_weights":RegA_weights,"RegA_p":RegA_p,"RegA_reweighted":RegA_reweighted
                            ,"RegC_ht":RegC_ht,"RegC_weights":RegC_weights,"RegC_p":RegC_p
                            ,"RegB_ht":RegB_ht,"RegB_weights":RegB_weights,"RegB_p":RegB_p,"RegB_reweighted":RegB_reweighted
                            ,"RegD_ht":RegD_ht,"RegD_weights":RegD_weights,"RegD_p":RegD_p})
    return RegA_p #, RegB_p  # return predicted values for CI tests

def options():
    parser = argparse.ArgumentParser()
    # input files d
    parser.add_argument("-c",  "--conf", help="Configuration file. If provided, all other settings are overruled.", default=None)
    parser.add_argument("-i",  "--inFile", help="Input file.", default=None)
    parser.add_argument("-o",  "--outDir", help="Output directory", default="./")
    parser.add_argument("-m",  "--model_weights", help="Model weights.", default=None)
    return parser.parse_args()

if __name__ == "__main__":
    main()
