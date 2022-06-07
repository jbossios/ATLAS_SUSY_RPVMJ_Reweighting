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
import os
import gc
from glob import glob

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

def main():

    # user options
    ops = options()

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
                # np.array(f['EventVars']['deta']),
                # np.array(f['EventVars']['djmass']),
                # np.array(f['EventVars']['minAvgMass']),
                # np.array(f['source']['pt'][:,0])
           ],-1)
        HT = np.array(f['EventVars']['HT'])
        minAvgMass = np.array(f['EventVars']['minAvgMass'])
        nQuarkJets = (np.array(f['source']['QGTaggerBDT']) > cut_QGTaggerBDT).sum(1)
        normweight = np.array(f['normweight']['normweight'])
        print(f"Number of events: {minAvgMass.shape[0]}")

    # Create cuts to Reweight A -> C
    RegA = np.logical_and(minAvgMass < cut_minAvgMass, nQuarkJets < cut_nQuarkJets)
    RegC = np.logical_and(minAvgMass < cut_minAvgMass, nQuarkJets >= cut_nQuarkJets)
    RegB = np.logical_and(minAvgMass >= cut_minAvgMass, nQuarkJets < cut_nQuarkJets)
    RegD = np.logical_and(minAvgMass >= cut_minAvgMass, nQuarkJets >= cut_nQuarkJets)
    print(f"Number of events in A and C: {RegA.sum()}, {RegC.sum()}")
    print(f"Number of events in B and D: {RegB.sum()}, {RegD.sum()}")
    del minAvgMass, nQuarkJets
    gc.collect()

    # pickpup weights
    RegA_weights = normweight[RegA]
    RegC_weights = normweight[RegC]
    RegB_weights = normweight[RegB]
    RegD_weights = normweight[RegD]
    del normweight
    gc.collect()

    # get events per region
    RegA_ht = x[:,0][RegA] 
    RegC_ht = x[:,0][RegC]
    RegB_ht = x[:,0][RegB]
    RegD_ht = x[:,0][RegD]
    del x
    gc.collect()

    # load predictions
    pred_files = handleInput(ops.predFile)

    for iP, pred_file in enumerate(pred_files):

        with np.load(pred_file, allow_pickle=True) as preds:
            # reweight
            RegA_reweighted = RegA_weights * np.exp(preds["RegA_p"])
            RegB_reweighted = RegB_weights * np.exp(preds["RegB_p"])

        # Reweight A -> C
        fig, [ax,rx] = plt.subplots(2,1,constrained_layout=False,sharey=False,sharex=True,gridspec_kw={"height_ratios": [3.5,1], 'hspace':0.0},)
        rx.set_ylabel("Ratio\nTo RegC")
        rx.set_xlabel(r"H$_{\mathrm{T}}$ [GeV]")
        rx.set_ylim(0,2)
        ax.set_ylabel("Density of Events")
        ax.set_yscale("log")
        bins = np.linspace(0, 13000, 100)
        c0, bin_edges, _ = ax.hist(RegA_ht, bins = bins, weights = RegA_weights, label = rf'RegA NQuarkJets $<$ {cut_nQuarkJets}', color = colors[0], density=ops.density, histtype="step", lw=2)
        c1, bin_edges, _ = ax.hist(RegC_ht, bins = bins, weights = RegC_weights, label = rf'RegC NQuarkJets $\geq$ {cut_nQuarkJets}', color = colors[1], density=ops.density, histtype="step", lw=2)
        c2, bin_edges, _ = ax.hist(RegA_ht, bins = bins, weights = RegA_reweighted, label = rf'Reweight RegA $\rightarrow$ RegC', color = colors[2], density=ops.density, histtype="step", lw=2) 
        rx.plot((bin_edges[:-1] + bin_edges[1:]) / 2, c0/(c1 + 10**-50), 'o-', label = rf'RegA $/$ RegC', color = colors[0], lw=1)
        rx.plot((bin_edges[:-1] + bin_edges[1:]) / 2, c2/(c1 + 10**-50), 'o-', label = rf'Reweighted RegA $/$ RegC', color = colors[2], lw=1)
        rx.plot((bin_edges[:-1] + bin_edges[1:]) / 2,[1] * len((bin_edges[:-1] + bin_edges[1:]) / 2), ls="--",color="black",alpha=0.8)
        ax.legend(title=rf"minAvgMass $<$ {cut_minAvgMass} GeV", loc="best", prop={'size': 8}, framealpha=0.0)
        # rx.legend(title="", loc="best", prop={'size': 7}, framealpha=0.0)
        plt.savefig(os.path.join(ops.outDir,f'reweightAtoC_bootstrap{iP}.pdf'), bbox_inches="tight")

        # Reweight B -> D
        fig, [ax,rx] = plt.subplots(2,1,constrained_layout=False,sharey=False,sharex=True,gridspec_kw={"height_ratios": [3.5,1], 'hspace':0.0},)
        rx.set_ylabel("Ratio\nTo RegD")
        rx.set_xlabel(r"H$_{\mathrm{T}}$ [GeV]")
        rx.set_ylim(0,2)
        ax.set_ylabel("Density of Events")
        ax.set_yscale("log")
        bins = np.linspace(0, 13000, 100)
        c0, bin_edges, _ = ax.hist(RegB_ht, bins = bins, weights = RegB_weights, label = rf'RegB NQuarkJets $<$ {cut_nQuarkJets}', color = colors[0], density=ops.density, histtype="step", lw=2)
        c1, bin_edges, _ = ax.hist(RegD_ht, bins = bins, weights = RegD_weights, label = rf'RegD NQuarkJets $\geq$ {cut_nQuarkJets}', color = colors[1], density=ops.density, histtype="step", lw=2)
        c2, bin_edges, _ = ax.hist(RegB_ht, bins = bins, weights = RegB_reweighted, label = rf'Reweight RegB $\rightarrow$ RegD', color = colors[2], density=ops.density, histtype="step", lw=2) 
        rx.plot((bin_edges[:-1] + bin_edges[1:]) / 2, c0/(c1 + 10**-50), 'o-', label = rf'RegB $/$ RegD', color = colors[0], lw=1)
        rx.plot((bin_edges[:-1] + bin_edges[1:]) / 2, c2/(c1 + 10**-50), 'o-', label = rf'Reweighted RegB $/$ RegD', color = colors[2], lw=1)
        rx.plot((bin_edges[:-1] + bin_edges[1:]) / 2,[1] * len((bin_edges[:-1] + bin_edges[1:]) / 2), ls="--",color="black",alpha=0.8)
        ax.legend(title=rf"minAvgMass $\geq$ {cut_minAvgMass} GeV", loc="best", prop={'size': 8}, framealpha=0.0)
        # rx.legend(title="", loc="best", prop={'size': 7}, framealpha=0.0)
        plt.savefig(os.path.join(ops.outDir,f'reweightBtoD_bootstrap{iP}.pdf'), bbox_inches="tight")

def options():
    parser = argparse.ArgumentParser()
    # input files d
    parser.add_argument("-c",  "--conf", help="Configuration file. If provided, all other settings are overruled.", default=None)
    parser.add_argument("-i",  "--inFile", help="Input file.", default=None)
    parser.add_argument("-p",  "--predFile", help="Prediction file.", default=None)
    parser.add_argument("-o",  "--outDir", help="Output directory", default="./")
    parser.add_argument("-m",  "--model_weights", help="Model weights.", default=None)
    parser.add_argument("-d", "--density", help="Make plots density=True", action="store_true")
    return parser.parse_args()

def handleInput(data):
    elif os.path.isfile(data) and ".npz" in os.path.basename(data):
        return [data]
    elif os.path.isfile(data) and ".txt" in os.path.basename(data):
        return sorted([line.strip() for line in open(data,"r")])
    elif os.path.isdir(data):
        return sorted([os.path.join(data,i) for i in os.listdir(data)])
    elif "*" in data:
        return sorted(glob(data))
    return []

if __name__ == "__main__":
    main()
