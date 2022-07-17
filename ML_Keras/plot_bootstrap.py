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
import scipy.stats
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

    # control and validation regions
    cut_deta12 = 1.5
    CR_njets, VR_njets, SR_njets = 5, 6, 7
    CR_high = np.logical_and(dEta12 >= cut_deta12, n_jets == CR_njets)
    CR_low  = np.logical_and(dEta12 < cut_deta12,  n_jets == CR_njets)
    VR_high = np.logical_and(dEta12 >= cut_deta12, n_jets == VR_njets)
    VR_low  = np.logical_and(dEta12 < cut_deta12,  n_jets == VR_njets)
    SR_high = np.logical_and(dEta12 >= cut_deta12, n_jets >= SR_njets)
    SR_low  = np.logical_and(dEta12 < cut_deta12,  n_jets >= SR_njets)
    R = {
        "CR_high" : CR_high,
        "CR_low" : CR_low,
        "VR_high" : VR_high,
        "VR_low" : VR_low,
        "SR_high" : SR_high,
        "SR_low" : SR_low
    }

    # store variables
    var = {
        "HT" : [HT, r"H$_{\mathrm{T}}$ [GeV]", np.linspace(cut_HT,13000,100)],
        "minAvg" : [minAvg, "minAvgMass [GeV]", np.linspace(cut_minAvg,3500,100)],
        #"dEta12" : [dEta12, "dEta12", np.linspace(0,4,20)],
        #"n_jets" : [n_jets, "nJets", np.linspace(0,20,21)],
        "djmass" : [djmass, "djmass", np.linspace(0,15000,100)]
    }

    # load predictions
    pred_files = handleInput(ops.predFile)
    y_label = "Density of Events" if ops.density else "Number of Events"
    # loop over variables
    for key, [x, xlabel, bins] in var.items():

        # loop over regions
        for region in ["CR", "VR", "SR"]:

            # loop over trainings
            bootstrap, hists = [], []
            for iP, pred_file in enumerate(pred_files):

                print(f"Plotting bootstrap {iP}/{len(pred_files)}: {pred_file}")

                # load reweighting
                with h5py.File(pred_file, "r") as w:
                    reweight = np.exp(w[f'{region}_high_p'])

                high_x, high_w = x[R[f"{region}_high"]], normweight[R[f"{region}_high"]]
                low_x, low_w = x[R[f"{region}_low"]], normweight[R[f"{region}_low"]]

                # Control Region
                fig, [ax,rx] = plt.subplots(2,1,constrained_layout=False,sharey=False,sharex=True,gridspec_kw={"height_ratios": [3.5,1], 'hspace':0.0},)
                rx.set_ylabel("Ratio\nTo Low")
                rx.set_xlabel(xlabel)
                rx.set_ylim(0,2)
                ax.set_ylabel(y_label)
                ax.set_yscale("log")
                c0, bin_edges, _ = ax.hist(high_x, bins = bins, weights = high_w, label = rf'High', color = colors[0], density=ops.density, histtype="step", lw=2)
                c1, bin_edges, _ = ax.hist(low_x,  bins = bins, weights = low_w, label = rf'Low', color = colors[1], density=ops.density, histtype="step", lw=2)
                c2, bin_edges, _ = ax.hist(high_x, bins = bins, weights = high_w*reweight, label = rf'Reweight High $\rightarrow$ Low', color = colors[2], density=ops.density, histtype="step", lw=2) 

                rx.plot((bin_edges[:-1] + bin_edges[1:]) / 2, c0/(c1 + 10**-50), 'o-', label = rf'High $/$ Low', color = colors[0], lw=1)
                rx.plot((bin_edges[:-1] + bin_edges[1:]) / 2, c2/(c1 + 10**-50), 'o-', label = rf'Reweighted High $/$ Low', color = colors[2], lw=1)
                rx.plot((bin_edges[:-1] + bin_edges[1:]) / 2,[1] * len((bin_edges[:-1] + bin_edges[1:]) / 2), ls="--",color="black",alpha=0.8)
                
                ax.legend(title=rf"{region}", loc="best", prop={'size': 8}, framealpha=0.0)
                # rx.legend(title="", loc="best", prop={'size': 7}, framealpha=0.0)
                plt.savefig(os.path.join(ops.outDir,f'bootstrap{iP}_{region}_{key}.pdf'), bbox_inches="tight")
                plt.close(fig)
                
                bootstrap.append(high_w*reweight)
                hists.append(c2)
            
            # produce bootstrap plot
            # temp = np.stack(bootstrap,0)
            # iqr = scipy.stats.iqr(temp,0)
            # median = np.median(temp,0)
            # w_nom = median
            # w_up = median + 0.5*iqr
            # w_down = median - 0.5*iqr

            # # plot
            # fig, [ax,rx] = plt.subplots(2,1,constrained_layout=False,sharey=False,sharex=True,gridspec_kw={"height_ratios": [3.5,1], 'hspace':0.0},)
            # rx.set_ylabel("Ratio")
            # rx.set_xlabel(xlabel)
            # rx.set_ylim(0,2)
            # ax.set_ylabel(y_label)
            # ax.set_yscale("log")
            # c0, bin_edges, _ = ax.hist(high_x, bins = bins, weights = high_w, label = rf'High', color = colors[0], density=ops.density, histtype="step", lw=2)
            # c1, bin_edges, _ = ax.hist(low_x,  bins = bins, weights = low_w, label = rf'Low', color = colors[1], density=ops.density, histtype="step", lw=2)
            
            # h_nom, bin_edges, _  = ax.hist(high_x, bins = bins, weights = w_nom, label = rf'Median Reweight High $\rightarrow$ Low', color = colors[2], density=ops.density, histtype="step", lw=2) 
            # h_up, bin_edges, _   = ax.hist(high_x, bins = bins, weights = w_up, label = rf'Up Reweight High $\rightarrow$ Low', color = colors[3], density=ops.density, histtype="step", lw=2) 
            # h_down, bin_edges, _ = ax.hist(high_x, bins = bins, weights = w_down, label = rf'Down Reweight High $\rightarrow$ Low', color = colors[4], density=ops.density, histtype="step", lw=2) 

            # rx.plot((bin_edges[:-1] + bin_edges[1:]) / 2, c0/(c1 + 10**-50), 'o-', label = rf'High $/$ Low', color = colors[0], lw=1)
            # rx.plot((bin_edges[:-1] + bin_edges[1:]) / 2, h_nom/(c1 + 10**-50), 'o-', label = rf'Median Reweighted High $/$ Low', color = colors[2], lw=1)
            # rx.plot((bin_edges[:-1] + bin_edges[1:]) / 2, h_up/(c1 + 10**-50), 'o-', label = rf'Up Reweighted High $/$ Low', color = colors[3], lw=1)
            # rx.plot((bin_edges[:-1] + bin_edges[1:]) / 2, h_down/(c1 + 10**-50), 'o-', label = rf'Down Reweighted High $/$ Low', color = colors[4], lw=1)

            # rx.plot((bin_edges[:-1] + bin_edges[1:]) / 2,[1] * len((bin_edges[:-1] + bin_edges[1:]) / 2), ls="--",color="black",alpha=0.8)
            # ax.legend(title=rf"{region}", loc="best", prop={'size': 8}, framealpha=0.0)
            # # rx.legend(title="", loc="best", prop={'size': 7}, framealpha=0.0)
            # plt.savefig(os.path.join(ops.outDir,f'bootstrap{iP}_{region}_{key}_w_nom.pdf'), bbox_inches="tight")
            # plt.close(fig)

            # # compare bootstraps
            # fig, ax = plt.subplots(1,  gridspec_kw={'height_ratios': (1,), 'hspace': 0.0})
            # ax.plot((bin_edges[:-1] + bin_edges[1:]) / 2, [1] * len((bin_edges[:-1] + bin_edges[1:]) / 2), ls='--', color = 'black', lw=1)
            # for iH, hist in enumerate(hists):
            #     if iH == 0:
            #         ax.plot((bin_edges[:-1] + bin_edges[1:]) / 2, hist/(h_nom + 10**-50), '-', color = 'grey', lw=1, label="Single Bootstraped Estimate")
            #     else:
            #         ax.plot((bin_edges[:-1] + bin_edges[1:]) / 2, hist/(h_nom + 10**-50), '-', color = 'grey', lw=1)
            # ax.plot((bin_edges[:-1] + bin_edges[1:]) / 2, h_up/(h_nom + 10**-50), '-', color = 'blue', lw=1, label=r"Vary All Event Weights Up/Down ($\pm$ IQR/2)")
            # ax.plot((bin_edges[:-1] + bin_edges[1:]) / 2, h_down/(h_nom + 10**-50), '-', color = 'blue', lw=1)
            # ax.set_ylim(0,2)
            # ax.set_ylabel("Bootstrap/Nominal Est.")
            # ax.set_xlabel(xlabel)
            # ax.legend(title=rf"Reweight High $\rightarrow$ Low", loc="best", prop={'size': 8}, framealpha=0.0)
            # plt.savefig(os.path.join(ops.outDir,f'bootstrap{iP}_{region}_{key}_div_non.pdf'), bbox_inches="tight")
            # plt.close(fig)


def options():
    parser = argparse.ArgumentParser()
    # input files d
    parser.add_argument("-c",  "--conf", help="Configuration file. If provided, all other settings are overruled.", default=None)
    parser.add_argument("-i",  "--inFile", help="Input file.", default=None)
    parser.add_argument("-p",  "--predFile", help="Prediction file.", default=None)
    parser.add_argument("-o",  "--outDir", help="Output directory", default="./")
    parser.add_argument("-d", "--density", help="Make plots density=True", action="store_true")
    return parser.parse_args()

def handleInput(data):
    if os.path.isfile(data) and ".h5" in os.path.basename(data):
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
