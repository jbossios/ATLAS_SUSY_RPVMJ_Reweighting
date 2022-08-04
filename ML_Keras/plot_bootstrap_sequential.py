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
        x = np.stack([
                np.array(f['EventVars']['HT']),
                np.array(f['EventVars']['deta']),
                np.array(f['EventVars']['djmass']),
                np.array(f['EventVars']['minAvgMass']),
                np.array(f['source']['pt'][:,0]),
                np.array(f['source']['pt'][:,3]),
                np.array(f['source']['pt'][:,5]),
           ],-1)
        deta = np.array(f['EventVars']['deta'])
        minAvgMass = np.array(f['EventVars']['minAvgMass'])
        nQuarkJets = (np.array(f['source']['QGTaggerBDT']) > cut_QGTaggerBDT).sum(1)
        normweight = np.array(f['normweight']['normweight'])
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

    # pickpup weights
    Reg0qincl_weights = normweight[Reg0qincl]
    Reg1qincl_weights = normweight[Reg1qincl]
    Reg1qCR_weights = normweight[Reg1qCR]
    Reg2qCR_weights = normweight[Reg2qCR]
    Reg1qSR_weights = normweight[Reg1qSR]
    Reg2qSR_weights = normweight[Reg2qSR]
    del normweight
    gc.collect()


    # load predictions
    pred_files = handleInput(ops.predFile)

    regions = {
        "Reg0qincl":"Reg1qincl",
        "Reg1qCR":"Reg2qCR",
        "Reg1qSR":"Reg2qSR",
    }
    for i, var in enumerate(('HT','deta','djmass','minavg','pt0','pt3','pt5')):
        # get events per region
        Reg0qincl_x = x[:,i][Reg0qincl] 
        Reg1qincl_x = x[:,i][Reg1qincl]
        Reg1qCR_x = x[:,i][Reg1qCR] 
        Reg2qCR_x = x[:,i][Reg2qCR]
        Reg1qSR_x = x[:,i][Reg1qSR]
        Reg2qSR_x = x[:,i][Reg2qSR]
        # store reweighting
        reweightings = {k: [] for k in regions}
        hists        = {k: [] for k in regions}

        for iP, pred_file in enumerate(pred_files):
            print(f"Plotting {var}, bootstrap {iP}/{len(pred_files)}: {pred_file}")

            with np.load(pred_file, allow_pickle=True) as preds:
                # reweight
                try:
                    Reg0qincl_reweighted = Reg0qincl_weights * np.exp(preds["Reg0qincl_p"])
                    Reg1qCR_reweighted = Reg1qCR_weights * np.exp(preds["Reg1qCR_p"])
                    Reg1qSR_reweighted = Reg1qSR_weights * np.exp(preds["Reg1qSR_p"])
                except KeyError:
                    Reg0qincl_reweighted = Reg0qincl_weights * np.exp(preds["Reg0q_incl_p"])
                    Reg1qCR_reweighted = Reg1qCR_weights * np.exp(preds["Reg1q_CR_p"])
                    Reg1qSR_reweighted = Reg1qSR_weights * np.exp(preds["Reg1q_SR_p"])

            # store the reweighting
            reweightings["Reg0qincl"].append(Reg0qincl_reweighted)
            reweightings["Reg1qCR"].append(Reg1qCR_reweighted)
            reweightings["Reg1qSR"].append(Reg1qSR_reweighted)

            h = plot_individual(var, ops, iP, "Reg0qincl", "Reg1qincl", Reg0qincl_x, Reg1qincl_x, Reg0qincl_weights, Reg1qincl_weights, Reg0qincl_reweighted)
            hists["Reg0qincl"].append(h)
            h = plot_individual(var, ops, iP, "Reg1qCR", "Reg2qCR", Reg1qCR_x, Reg2qCR_x, Reg1qCR_weights, Reg2qCR_weights, Reg1qCR_reweighted)
            hists["Reg1qCR"].append(h)
            h = plot_individual(var, ops, iP, "Reg1qSR", "Reg2qSR", Reg1qSR_x, Reg2qSR_x, Reg1qSR_weights, Reg2qSR_weights, Reg1qSR_reweighted)
            hists["Reg1qSR"].append(h)

        # concatenate and take median +- 1/2 * IQR
        bootstrap_weights = {}
        for key, val in reweightings.items():
            ensemble = np.stack(val,0)
            iqr = scipy.stats.iqr(ensemble,0)
            median = np.median(ensemble,0)
            bootstrap_weights[key] = {
                'w_nom' : median,
                'w_up' : median + 0.5 * iqr,
                'w_down' : median - 0.5 * iqr   
            }

        plot_ensemble(var, ops, "Reg0qincl", "Reg1qincl", Reg0qincl_x, Reg1qincl_x, Reg0qincl_weights, Reg1qincl_weights, hists, bootstrap_weights)
        plot_ensemble(var, ops, "Reg1qCR", "Reg2qCR", Reg1qCR_x, Reg2qCR_x, Reg1qCR_weights, Reg2qCR_weights, hists, bootstrap_weights)
        plot_ensemble(var, ops, "Reg1qSR", "Reg2qSR", Reg1qSR_x, Reg2qSR_x, Reg1qSR_weights, Reg2qSR_weights, hists, bootstrap_weights)

def plot_individual(var, ops, iP, source_name, target_name, source_var, target_var, source_weights, target_weights, source_reweighted):

    bins = get_binning(var, ops)

    # Reweight source -> target
    fig, [ax,rx] = plt.subplots(2,1,constrained_layout=False,sharey=False,sharex=True,gridspec_kw={"height_ratios": [3.5,1], 'hspace':0.0},)
    rx.set_ylabel(f"Ratio\nTo {target_name}")
    rx.set_xlabel(get_axis_label(var))
    rx.set_ylim(0,2)
    ax.set_ylabel("Density of Events")
    ax.set_yscale("log")
    c0, bin_edges, _ = ax.hist(source_var, bins = bins, weights = source_weights,    label = get_label(source_name), color = colors[0], density=ops.density, histtype="step", lw=2)
    c1, bin_edges, _ = ax.hist(target_var, bins = bins, weights = target_weights,    label = get_label(target_name), color = colors[1], density=ops.density, histtype="step", lw=2)
    c2, bin_edges, _ = ax.hist(source_var, bins = bins, weights = source_reweighted, label = get_label(source_name, target_name), color = colors[2], density=ops.density, histtype="step", lw=2) 
    rx.plot((bin_edges[:-1] + bin_edges[1:]) / 2, c0/(c1 + 10**-50), 'o-', label = rf'{source_name} / {target_name}', color = colors[0], lw=1)
    rx.plot((bin_edges[:-1] + bin_edges[1:]) / 2, c2/(c1 + 10**-50), 'o-', label = rf'Reweighted {source_name} / {target_name}', color = colors[2], lw=1)
    rx.plot((bin_edges[:-1] + bin_edges[1:]) / 2,[1] * len((bin_edges[:-1] + bin_edges[1:]) / 2), ls="--",color="black",alpha=0.8)
    ax.legend(title=get_title(source_name, ops), loc="best", prop={'size': 8}, framealpha=0.0)
    # rx.legend(title="", loc="best", prop={'size': 7}, framealpha=0.0)
    plt.savefig(os.path.join(ops.outDir,f'reweight{var}_{source_name}to{target_name}_bootstrap{iP}.pdf'), bbox_inches="tight")
    plt.close(fig)
    return c2


def plot_ensemble(var, ops, source_name, target_name, source_var, target_var, source_weights, target_weights, hists, bootstrap_weights):

    bins = get_binning(var, ops)

    # plot
    fig, [ax,rx] = plt.subplots(2,1,constrained_layout=False,sharey=False,sharex=True,gridspec_kw={"height_ratios": [3.5,1], 'hspace':0.0},)
    rx.set_ylabel(f"Ratio\nTo {target_name}")
    rx.set_xlabel(get_axis_label(var))
    rx.set_ylim(0,2)
    ax.set_ylabel("Density of Events")
    ax.set_yscale("log")
    c0, bin_edges, _ = ax.hist(source_var, bins = bins, weights = source_weights, label = get_label(source_name), color = colors[0], density=ops.density, histtype="step", lw=2)
    c1, bin_edges, _ = ax.hist(target_var, bins = bins, weights = target_weights, label = get_label(target_name), color = colors[1], density=ops.density, histtype="step", lw=2)
    c2, bin_edges, _ = ax.hist(source_var, bins = bins, weights = bootstrap_weights[source_name]["w_nom"],  label = rf'Median '+get_label(source_name, target_name), color = colors[2], density=ops.density, histtype="step", lw=2) 
    c3, bin_edges, _ = ax.hist(source_var, bins = bins, weights = bootstrap_weights[source_name]["w_up"],   label = rf'Up '+get_label(source_name, target_name), color = colors[3], density=ops.density, histtype="step", lw=2) 
    c4, bin_edges, _ = ax.hist(source_var, bins = bins, weights = bootstrap_weights[source_name]["w_down"], label = rf'Down '+get_label(source_name, target_name), color = colors[4], density=ops.density, histtype="step", lw=2) 
    rx.plot((bin_edges[:-1] + bin_edges[1:]) / 2, c0/(c1 + 10**-50), 'o-', label = rf'{source_name} $/$ {target_name}', color = colors[0], lw=1)

    if ops.band:
        rx.fill_between((bin_edges[:-1] + bin_edges[1:]) / 2, c4/(c1 + 10**-50), c3/(c1 + 10**-50), alpha=0.2, edgecolor = colors[3], facecolor = colors[3], lw=1, label = 'Interquartile')
    else:
        rx.plot((bin_edges[:-1] + bin_edges[1:]) / 2, c3/(c1 + 10**-50), 'o-', label = rf'Up Reweighted {source_name} $/$ {target_name}', color = colors[3], lw=1)
        rx.plot((bin_edges[:-1] + bin_edges[1:]) / 2, c4/(c1 + 10**-50), 'o-', label = rf'Down Reweighted {source_name} $/$ {target_name}', color = colors[4], lw=1)
    rx.plot((bin_edges[:-1] + bin_edges[1:]) / 2,[1] * len((bin_edges[:-1] + bin_edges[1:]) / 2), ls="--",color="black",alpha=0.8)
    ax.legend(title=get_title(source_name, ops), loc="best", prop={'size': 8}, framealpha=0.0)
    # rx.legend(title="", loc="best", prop={'size': 7}, framealpha=0.0)
    plt.savefig(os.path.join(ops.outDir,f'reweight{var}_{source_name}to{target_name}_bootstrap_w_nom.pdf'), bbox_inches="tight")
    plt.close(fig)

    # compare bootstraps
    fig, ax = plt.subplots(1,  gridspec_kw={'height_ratios': (1,), 'hspace': 0.0})
    ax.plot((bin_edges[:-1] + bin_edges[1:]) / 2, [1] * len((bin_edges[:-1] + bin_edges[1:]) / 2), ls='--', color = 'black', lw=1)
    for iH, hist in enumerate(hists[source_name]):
        if iH == 0:
            ax.plot((bin_edges[:-1] + bin_edges[1:]) / 2, hist/(c2 + 10**-50), '-', color = 'grey', lw=1, label="Single Bootstraped Estimate")
        else:
            ax.plot((bin_edges[:-1] + bin_edges[1:]) / 2, hist/(c2 + 10**-50), '-', color = 'grey', lw=1)
    ax.plot((bin_edges[:-1] + bin_edges[1:]) / 2, c3/(c2 + 10**-50), '-', color = 'blue', lw=1, label=r"Vary All Event Weights Up/Down ($\pm$ IQR/2)")
    ax.plot((bin_edges[:-1] + bin_edges[1:]) / 2, c4/(c2 + 10**-50), '-', color = 'blue', lw=1)
    ax.set_ylim(0,2)
    ax.set_ylabel("Bootstrap/Nominal Est.")
    ax.set_xlabel(get_axis_label(var))
    ax.legend(title=rf"Reweight {source_name} $\rightarrow$ {target_name}", loc="best", prop={'size': 8}, framealpha=0.0)
    plt.savefig(os.path.join(ops.outDir,f'reweight{source_name}to{target_name}_bootstrap_div_nominal.pdf'), bbox_inches="tight")
    plt.close(fig)

def get_axis_label(var):
    if var == 'HT':
        return r"H$_{\mathrm{T}}$ [GeV]"
    if var == 'minavg':
        return r"min avg mass [GeV]"
    if var == 'deta':
        return r"$\Delta\eta$"
    if var == 'djmass':
        return r"$m_{jj}$"
    if var == 'pt0':
        return r"Leading jet $p_T$"
    if var == 'pt3':
        return r"Fourth jet $p_T$"
    if var == 'pt5':
        return r"Sixth jet $p_T$"
    return 'Define me'

def get_label(reg, target=None):
    if reg == "Reg0qincl":
        label = '0 quarks inclusive'
        if target:
            label = r'0 quarks $\rightarrow$ 1 quark inclusive'
    elif reg == "Reg1qincl":
        label = '1 quark inclusive'
    elif reg == "Reg1qCR":
        label = '1 quark CR'
        if target:
            label = r'1 quark $\rightarrow$ 2 quarks CR'
    elif reg == "Reg2qCR":
        label = '2 quarks CR'
    elif reg == "Reg1qSR":
        label = '1 quark SR'
        if target:
            label = r'1 quark $\rightarrow$ 2 quarks SR'
    elif reg == "Reg2qSR":
        label = '2 quarks SR'
    else:
        print("WTF",reg)
    return label

def get_title(source, ops):
    if source == "Reg0qincl":
        label = 'Inclusive selection'
    elif source == "Reg1qCR":
        label = r'minAvgMass $<$ 750 GeV'
        if ops.SR2D:
            label = r'minAvgMass $<$ 750 GeV || $\Delta\eta > 1.4$'
    elif source == "Reg1qSR":
        label = r'minAvgMass $\geq$ 750 GeV'
        if ops.SR2D:
            label = r'minAvgMass $\geq$ 750 GeV && $\Delta\eta < 1.4$'
    return label

def get_binning(var, ops):
    allow_irregular = True
    if var == 'HT':
        xmin, xmax = 300, 13000
    if var == 'minavg':
        xmin, xmax = 100, 4000
    if var == 'deta':
        xmin, xmax = -1, 5
        allow_irregular = False
    if var == 'djmass':
        xmin, xmax = 0, 4000
        allow_irregular = False
    if var == 'pt0':
        xmin, xmax = 10, 4000
    if var == 'pt3':
        xmin, xmax = 10, 2000
    if var == 'pt5':
        xmin, xmax = 10, 700

    if ops.irregular_binning and allow_irregular:
        return np.geomspace(xmin, xmax, 100)
    else:
        return np.linspace(0, xmax, 100)

def options():
    parser = argparse.ArgumentParser()
    # input files d
    parser.add_argument("-c",  "--conf", help="Configuration file. If provided, all other settings are overruled.", default=None)
    parser.add_argument("-i",  "--inFile", help="Input file.", default="/eos/user/k/kbai/h5_forJavier/user.abadea.DijetsALL_minJetPt50_minNjets6_maxNjets8_v1.h5")
    parser.add_argument("-p",  "--predFile", help="Prediction file.", default=None)
    parser.add_argument("-o",  "--outDir", help="Output directory", default="./")
    parser.add_argument("-m",  "--model_weights", help="Model weights.", default=None)
    parser.add_argument("-nd", "--no-density", help="Make plots density=False", action="store_true")
    parser.add_argument("-f", "--folder", help="Read and store everything from this folder")
    parser.add_argument("--SR2D", action="store_true", help="Define 2D SR")
    parser.add_argument("--irregular-binning", action="store_true", help="Use wider bins at high HT")
    parser.add_argument("--band", action="store_true", help="Show uncertainty band")

    args = parser.parse_args()
    args.density = not args.no_density

    if args.folder:
        args.predFile = os.path.join(args.folder, 'training_*/*npz')
        args.model_weights = os.path.join(args.folder, 'training_*')
        args.outDir = args.folder

    return args

def handleInput(data):
    if os.path.isfile(data) and ".npz" in os.path.basename(data):
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
