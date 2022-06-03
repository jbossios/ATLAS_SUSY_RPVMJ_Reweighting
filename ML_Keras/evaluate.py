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

# custom code
try:
    from get_data import get_full_data
    from make_model import make_model
except:
    from ML_Keras.get_data import get_full_data
    from ML_Keras.make_model import make_model

# Tensorflow GPU settings
physical_devices = tf.config.list_physical_devices('GPU') 
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# matplotlib
plt.rcParams['figure.figsize'] = (4,4)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True

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
                "input_dim" : ops.input_dim,
                "ndense" : ops.ndense,
                "nnode_per_dense" : ops.nnode_per_dense
            }

    # protection
    if ops.model_weights is None:
        log.error('ERROR: no model weights were provided, exiting')
        sys.exit(1)

    # load model
    model = make_model(input_dim=conf["input_dim"], ndense=conf["ndense"], nnode_per_dense=conf["nnode_per_dense"], learning_rate=1e-3)
    model.summary()
    # if checkpoint directory provided use the latest
    if os.path.isdir(ops.model_weights):
        latest = tf.train.latest_checkpoint(ops.model_weights)
        log.info(f"Using latest weights from checkpoint directory: {latest}")
        model.load_weights(latest).expect_partial()
    else:
        model.load_weights(ops.model_weights).expect_partial()
    # x, nQuarkJets, normweight = get_full_data(conf["file"])

    # # prepare data
    # x = x.reshape(x.size, 1)
    # nQuarkJets = nQuarkJets.reshape(x.size, 1)
    # normweight = normweight.reshape(x.size, 1)
    # xa = x[nQuarkJets>0]
    # xb = x[nQuarkJets==0]
    # bins = np.linspace(0, 8000, 100)
    # normweightsa = normweight[nQuarkJets>0]
    # normweightsb = normweight[nQuarkJets==0]

    # # make model prediction
    # p = model.predict(x)

    # plot
    # xa = np.multiply(xa, 1000)
    # xb = np.multiply(xb, 1000)
    # fig, [ax,rx] = plt.subplots(2,1,constrained_layout=False,sharey=False,sharex=True,figsize=(8, 8),gridspec_kw={"height_ratios": [3.5,1], 'hspace':0.0},)
    # rx.set_ylabel("Ratio")
    # rx.set_xlabel("HT [GeV]")
    # ax.set_ylabel("Density of Events")
    # c0, bin_edges, _ = ax.hist(xa, bins = bins, alpha = 0.5, weights = normweightsa, label = 'NQuarkJets > 0', color = 'red', density=True, histtype="step", lw=2)
    # c1, bin_edges, _ = ax.hist(xb, bins = bins, alpha = 0.5, weights = normweightsb, label = 'NQuarkJets = 0', color = 'blue', density=True, histtype="step", lw=2)
    # p = np.array(p).reshape(x.size, 1)
    # _pp = p[nQuarkJets>0]
    # final_weights = _pp/(1-_pp)
    # final_weights *= normweightsa
    # c2, bin_edges, _ = ax.hist(xa, bins = bins, alpha = 0.5, weights = final_weights, label = 'NQuarkJets > 0 reweighted to NQuarkJets = 0', color = 'yellow', density=True, histtype="step", lw=2) 
    # rx.plot((bin_edges[:-1] + bin_edges[1:]) / 2, c2/(c1 + 10**-50), 'o-', label = 'Reweighted NQuarkJets > 0 / NQuarkJets = 0', color = 'black', lw=1)
    # rx.plot((bin_edges[:-1] + bin_edges[1:]) / 2,[1] * len((bin_edges[:-1] + bin_edges[1:]) / 2), ls="--",color="gray",alpha=0.8)
    # rx.set_ylim(0,2)
    # ax.legend()
    # rx.legend()
    # ax.set_yscale("log")
    # plt.savefig(os.path.join(ops.outDir,'eval.pdf'))  # TODO: improve output name


    # make prediction
    f = h5py.File(conf["file"],"r")
    HT = np.array(f['EventVars']['HT'])
    p = model.predict(HT)
    print(np.mean(p),np.std(p))

    # cuts used
    cut_minAvgMass = 750
    cut_QGTaggerBDT = 0.0 
    cut_nQuarkJets = 2

    minAvgMass = np.array(f['EventVars']['minAvgMass'])
    nQuarkJets = (np.array(f['source']['QGTaggerBDT']) > cut_QGTaggerBDT).sum(1)
    normweight = np.array(f['normweight']['normweight'])

    # Reweight A -> C
    RegA = np.logical_and(minAvgMass < cut_minAvgMass, nQuarkJets < cut_nQuarkJets)
    RegC = np.logical_and(minAvgMass < cut_minAvgMass, nQuarkJets >= cut_nQuarkJets)

    # isolate region
    RegA_ht = HT[RegA]
    RegA_weights = normweight[RegA]
    RegC_ht = HT[RegC]
    RegC_weights = normweight[RegC]

    # compute reweighted
    RegA_p = p[RegA].flatten()
    RegA_reweighted = RegA_weights * RegA_p/(1-RegA_p + 10**-30)

    # plot
    fig, [ax,rx] = plt.subplots(2,1,constrained_layout=False,sharey=False,sharex=True,gridspec_kw={"height_ratios": [3.5,1], 'hspace':0.0},)
    rx.set_ylabel("Ratio")
    rx.set_xlabel(r"H$_{\mathrm{T}}$ [GeV]")
    rx.set_ylim(0,2)
    ax.set_ylabel("Density of Events")
    # ax.set_yscale("log")
    bins = np.linspace(7000, 13000, 20)
    c0, bin_edges, _ = ax.hist(RegA_ht, bins = bins, alpha = 0.5, weights = RegA_weights, label = rf'RegA NQuarkJets $<$ {cut_nQuarkJets}', color = 'red', density=True, histtype="step", lw=2)
    c1, bin_edges, _ = ax.hist(RegC_ht, bins = bins, alpha = 0.5, weights = RegC_weights, label = rf'RegC NQuarkJets $\geq$ {cut_nQuarkJets}', color = 'blue', density=True, histtype="step", lw=2)
    c2, bin_edges, _ = ax.hist(RegA_ht, bins = bins, alpha = 0.5, weights = RegA_reweighted, label = rf'Reweight RegA $\rightarrow$ RegC', color = 'yellow', density=True, histtype="step", lw=2) 
    rx.plot((bin_edges[:-1] + bin_edges[1:]) / 2, c0/(c1 + 10**-50), 'o-', label = rf'RegA $/$ RegC', color = 'gray', lw=1)
    rx.plot((bin_edges[:-1] + bin_edges[1:]) / 2, c2/(c1 + 10**-50), 'o-', label = rf'Reweighted RegA $/$ RegC', color = 'black', lw=1)
    rx.plot((bin_edges[:-1] + bin_edges[1:]) / 2,[1] * len((bin_edges[:-1] + bin_edges[1:]) / 2), ls="--",color="gray",alpha=0.8)
    ax.legend(title=rf"minAvgMass $<$ {cut_minAvgMass} GeV", loc="upper left", prop={'size': 8}, framealpha=0.0)
    rx.legend(title="", loc="best", prop={'size': 7}, framealpha=0.0)
    plt.savefig(os.path.join(ops.outDir,'reweightAtoC.pdf'), bbox_inches="tight")


    # Reweight B -> D
    RegB = np.logical_and(minAvgMass >= cut_minAvgMass, nQuarkJets < cut_nQuarkJets)
    RegD = np.logical_and(minAvgMass >= cut_minAvgMass, nQuarkJets >= cut_nQuarkJets)

    # isolate region
    RegB_ht = HT[RegB]
    RegB_weights = normweight[RegB]
    RegD_ht = HT[RegD]
    RegD_weights = normweight[RegD]

    # compute reweighted
    RegB_p = p[RegB].flatten()
    RegB_reweighted = RegB_weights * RegB_p/(1-RegB_p + 10**-30)

    # plot
    fig, [ax,rx] = plt.subplots(2,1,constrained_layout=False,sharey=False,sharex=True,gridspec_kw={"height_ratios": [3.5,1], 'hspace':0.0},)
    rx.set_ylabel("Ratio")
    rx.set_xlabel(r"H$_{\mathrm{T}}$ [GeV]")
    rx.set_ylim(0,2)
    ax.set_ylabel("Density of Events")
    # ax.set_yscale("log")
    bins = np.linspace(7000, 13000, 20)
    c0, bin_edges, _ = ax.hist(RegB_ht, bins = bins, alpha = 0.5, weights = RegB_weights, label = rf'RegB NQuarkJets $<$ {cut_nQuarkJets}', color = 'red', density=True, histtype="step", lw=2)
    c1, bin_edges, _ = ax.hist(RegD_ht, bins = bins, alpha = 0.5, weights = RegD_weights, label = rf'RegD NQuarkJets $\geq$ {cut_nQuarkJets}', color = 'blue', density=True, histtype="step", lw=2)
    c2, bin_edges, _ = ax.hist(RegB_ht, bins = bins, alpha = 0.5, weights = RegB_reweighted, label = rf'Reweight RegB $\rightarrow$ RegD', color = 'yellow', density=True, histtype="step", lw=2) 
    rx.plot((bin_edges[:-1] + bin_edges[1:]) / 2, c0/(c1 + 10**-50), 'o-', label = rf'RegB $/$ RegD', color = 'gray', lw=1)
    rx.plot((bin_edges[:-1] + bin_edges[1:]) / 2, c2/(c1 + 10**-50), 'o-', label = rf'Reweighted RegB $/$ RegD', color = 'black', lw=1)
    rx.plot((bin_edges[:-1] + bin_edges[1:]) / 2,[1] * len((bin_edges[:-1] + bin_edges[1:]) / 2), ls="--",color="gray",alpha=0.8)
    ax.legend(title=rf"minAvgMass $\geq$ {cut_minAvgMass} GeV", loc="upper left", prop={'size': 8}, framealpha=0.0)
    rx.legend(title="", loc="best", prop={'size': 7}, framealpha=0.0)
    plt.savefig(os.path.join(ops.outDir,'reweightBtoD.pdf'), bbox_inches="tight")

    return p  # return predicted values for CI tests

def options():
    parser = argparse.ArgumentParser()
    # input files d
    parser.add_argument("-c",  "--conf", help="Configuration file. If provided, all other settings are overruled.", default=None)
    parser.add_argument("-i",  "--inFile", help="Input file.", default=None)
    parser.add_argument("-o",  "--outDir", help="Output directory", default="./")
    parser.add_argument("-m",  "--model_weights", help="Model weights.", default=None)
    # model settings
    parser.add_argument("-ni", "--input_dim", help="Dimension of inputs per event for the first layer.", default=1, type=int)
    parser.add_argument("-nl", "--ndense", help="Number of dense layers.", default=1, type=int)
    parser.add_argument("-nd", "--nnode_per_dense", help="Number of nodes per dense layer.", default=30, type=int)
    return parser.parse_args()

if __name__ == "__main__":
    main()
