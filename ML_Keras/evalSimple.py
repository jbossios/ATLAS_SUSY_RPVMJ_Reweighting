'''
Authors: Anthony Badea, Jonathan Bossio
Date: Monday June 13, 2022
'''

# python imports
import h5py
import numpy as np
import argparse
import os
import uproot
import awkward as ak
import gc
from glob import glob

# custom code
try:
    from make_model import simple_model, sqrtR_loss, mean_pred
except:
    from ML_Keras.make_model import simple_model, sqrtR_loss, mean_pred

# Tensorflow GPU settings
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # CPU mode evaluation

# global variables
import logging
logging.basicConfig(format='%(levelname)s: %(message)s', level='INFO')
log = logging.getLogger('CreateH5Files')

# multiprocessing
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

def main():

    # user options
    ops = options() 
    
    # get file list
    input_files = handleInput(ops.inFile)

    # get list of ttrees using the first input file
    treeNames = ["trees_SRRPV_"]
    
    # get list of model weights
    if "training" in ops.model_weights:
        model_weights = [ops.model_weights]
    else:
        model_weights = handleInput(ops.model_weights)
    model_weights = [i for i in model_weights if "training" in i]
    print(model_weights)
    # make output dir
    if not os.path.isdir(ops.outDir):
        os.makedirs(ops.outDir)

    # create evaluation job dictionaries
    config  = []
    for inFileName in input_files:
        
        # create outfile tag
        tag = f"minJetPt{ops.minJetPt}_v{ops.version}"
        config.append({
            "inFileName" : inFileName,
            "treeNames" : treeNames,
            "tag" : tag,
            "model_weights" : model_weights
        })

    # launch jobs
    if ops.ncpu == 1:
        for conf in config:
            evaluate(conf)
    else:
        results = mp.Pool(ops.ncpu).map(evaluate, config)

def handleInput(data):
    # otherwise return 
    if os.path.isfile(data) and ".h5" in os.path.basename(data):
        return [data]
    elif os.path.isfile(data) and ".root" in os.path.basename(data):
        return [data]
    elif os.path.isfile(data) and ".txt" in os.path.basename(data):
        return sorted([line.strip() for line in open(data,"r")])
    elif os.path.isdir(data):
        return [os.path.join(data,i) for i in sorted(os.listdir(data))]
    elif "*" in data:
        return sorted(glob(data))
    return []

def evaluate(config):

    # user options
    ops = options()

    # make output file name
    outFileName = os.path.join(ops.outDir, os.path.basename(config["inFileName"]).strip(".root").strip(".h5") + f"_{config['tag']}_reweighting.h5")
    if os.path.isfile(outFileName) and not ops.doOverwrite:
        log.info(f"Skipping because output file already exists: {outFileName}")
        return

    # output data dict
    outData = {}

    # loop over trees
    for treeName in config["treeNames"]:

        # load data
        if os.path.basename(config["inFileName"]).endswith(".root"):
            with uproot.open(config["inFileName"]) as f:
                tree = f[treeName]

                if tree.num_entries == 0:
                    log.info(f"Skipping file because no entries: {config['inFileName']}")
                    return
                if "jet_e" not in tree.keys():
                    log.info(f"Skipping file because no key jet_e: {config['inFileName']}")
                    return

                # load kinematics for training
                kinem = {}
                jetBranches = ["jet_pt","jet_eta","jet_phi","jet_e"]
                for key in jetBranches:
                    kinem[key] = loadBranchAndPad(tree[key], -1)

                # make jet selections (zero out jets that fail)
                jet_selection = np.expand_dims(np.ones(kinem["jet_pt"].shape),-1)
                jet_selection = append_jet_selection(jet_selection, kinem["jet_pt"] >= ops.minJetPt)

                # apply final jet selection
                jet_selection = jet_selection.astype(bool)
                for key in kinem.keys():
                    kinem[key][~jet_selection[:,:,-1]] = 0

                # pick up event variables
                for key in ["minAvgMass_jetdiff10_btagdiff10"]:
                    kinem[key] = np.array(tree[key]).flatten()

                # compute variables
                kinem["jet_px"] = kinem["jet_pt"] * np.cos(kinem["jet_phi"])
                kinem["jet_py"] = kinem["jet_pt"] * np.sin(kinem["jet_phi"])
                kinem["jet_pz"] = kinem["jet_pt"] * np.sinh(kinem["jet_eta"])
                kinem["HT"] = kinem["jet_pt"].sum(1)
                kinem["deta"] = kinem["jet_eta"][:,0] - kinem["jet_eta"][:,1] # deta between leading jets
                kinem["djmass"] = np.sqrt(kinem["jet_e"][:,:2].sum(1)**2 - kinem["jet_px"][:,:2].sum(1)**2 - kinem["jet_py"][:,:2].sum(1)**2 - kinem["jet_pz"][:,:2].sum(1)**2) # mass of two leading jets added together
                
                # prepare input
                X = np.stack([
                    kinem["HT"],
                    kinem["deta"],
                    kinem["djmass"],
                    kinem["minAvgMass_jetdiff10_btagdiff10"],
                    kinem["jet_pt"][:,0]
                ], -1)

                # cleanup
                del kinem
                gc.collect()

        elif os.path.basename(config["inFileName"]).endswith(".h5"):
            # load file
            with h5py.File(config["inFileName"], "r") as f:
                # pick up variables from file
                X = np.stack([
                    np.array(f['EventVars']['HT']),
                    np.array(f['EventVars']['deta']),
                    np.array(f['EventVars']['djmass']),
                    np.array(f['EventVars']['minAvgMass']),
                    np.array(f['source']['pt'][:, 0])
                ], -1)

        # check number of entries
        if X.shape[0] == 0:
            log.info(f"Skipping file because X has no entries: {config['inFileName']}")

        # standardize input
        X = (X - np.mean(X,0))/np.std(X,0)

        # load model
        model = simple_model(input_dim=X.shape[1])
        model.compile(loss=sqrtR_loss, metrics=[mean_pred])
        # model.summary()

        # loop over model weights
        predictions = []
        for model_weights in config["model_weights"]:
            print(model_weights)
            # if checkpoint directory provided use the latest
            if os.path.isdir(model_weights):
                latest = tf.train.latest_checkpoint(model_weights)
                log.info(f"Using latest weights from checkpoint directory: {latest}")
                model.load_weights(latest).expect_partial()
            elif model_weights == "1":
                latest = tf.train.latest_checkpoint(glob.glob("checkpoints/*")[-1])
                log.info(f"Using latest weights from checkpoint directory: {latest}")
                model.load_weights(latest).expect_partial()
            else:
                model.load_weights(model_weights).expect_partial()

            # make prediction
            pred = model.predict(X, batch_size=40000).flatten()
            predictions.append(pred)

        # stack
        predictions = np.stack(predictions,-1)
        log.debug(f"Model prediction shape {predictions.shape}")
        outData[treeName] = {"reweighting": predictions}

    # save to file
    log.info(f"Saving: {outFileName}")
    with h5py.File(outFileName, 'w') as hf:
        for key, val in outData.items():
            Group = hf.create_group(key)
            for k, v in val.items():
                Group.create_dataset(k,data=v)
    log.info("Done!")


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inFile", help="Input file.", default=None, required=True)
    parser.add_argument("-o", "--outDir", help="Output directory", default="./")
    parser.add_argument("-w", "--model_weights", help="Model weights.", default=None, required=True)
    parser.add_argument('-v', '--version', default="0", help="Production version")
    parser.add_argument("-j",  "--ncpu", help="Number of cores to use for multiprocessing. If not provided multiprocessing not done.", default=1, type=int)
    parser.add_argument('--minJetPt', default=50, type=int, help="Minimum selected jet pt")
    parser.add_argument('--doOverwrite', action="store_true", help="Overwrite already existing files.")
    return parser.parse_args()


def loadBranchAndPad(branch, maxNjets, value=0):
    a = branch.array()
    a = ak.to_numpy(ak.fill_none(ak.pad_none(a, max(maxNjets,np.max(ak.num(a)))),value))
    return a

def append_jet_selection(original, new):
    return np.concatenate([original, np.expand_dims(np.logical_and(original[:,:,-1],new),-1)],-1)

if __name__ == "__main__":
    main()
