'''
Author: Anthony Badea
Date: May 21, 2022
Purpose: Sample dijet files based on event weights to create training files
'''

import numpy as np
import h5py
import argparse
import os
from multiprocessing import Pool

def main():

	ops = options()
	fileList = handleInput(ops.inFile)

	# load or make file with all weights included
	if ops.weightSampler:
	    x = np.load(ops.weightSampler,allow_pickle=True)
	    weights = x["weights"]
	    fileidx = x["fileidx"]
	else:
		weights = []
		fileidx = []
		for iF, file in enumerate(fileList):
		    print(f"File {iF}/{len(fileList)}")
		    with h5py.File(file, "r") as hf:
		        normweight = np.array(hf["normweight"]["normweight"]).flatten()
		        # print(normweight)
		        nevents = normweight.shape[0]
		        weights.append(normweight)
		        fileidx.append(np.stack([np.full((nevents),iF),np.arange(nevents)],-1))
		weights = np.concatenate(weights)
		fileidx = np.concatenate(fileidx)
		np.savez("WeightSamplerDijets.npz",**{"weights":weights,"fileidx":fileidx})

	probabilities = weights / weights.sum()


	n_files = int(ops.n_total_events/ops.n_events_per_file)
	configs = []
	for iF in range(n_files):
		configs.append({
			"probabilities" : probabilities, 
			"fileidx" : fileidx,
			"fileList" : fileList,
			"n_samples" : ops.n_events_per_file,
			"outName" : os.path.join(ops.outDir, f"Dijets_Sampled_v{ops.version}_{iF}.npz")
		})
	
	if ops.ncpu > 1:
		results = Pool(ncpu).map(add_normweight, configs)
	else:
		for conf in configs:
			sample(conf)

def sample(conf):
    # get sample list
    idx = np.random.choice(range(0,len(conf["probabilities"])), size=conf["n_samples"], p = conf["probabilities"], replace = True)
    samples = conf["fileidx"][idx]
    samples = samples[samples[:, 0].argsort()]
    # get unique files and list of events per file
    files = np.unique(samples[:,0])
    events = [samples[np.where(samples[:,0] == i)][:,1] for i in files]
    # create batch
    x = []
    y = []
    for iF,iE in zip(files,events):
        with h5py.File(conf["fileList"][iF], "r") as hf:
            # pick up the kinematics
            m = np.array(hf["source"]["mass"])
            pt = np.array(hf["source"]["pt"])
            eta = np.array(hf["source"]["eta"])
            phi = np.array(hf["source"]["phi"])
            j = np.stack([m,pt,eta,phi],-1)
            x.append(j[iE])
            y.append(np.zeros((j[iE].shape[0])))      
    x = np.concatenate(x)
    y = np.concatenate(y)

    # save in same format as spanet style
    np.savez(conf["outName"], **{"x":x,"y":y})

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inFile", help="Input file.", default=None)
    parser.add_argument("-o", "--outDir", help="Output directory", default="./")
    parser.add_argument("-w", "--weightSampler", help="Already made file to sample weights from. If not provided then one will be made automatically.", default=None)
    parser.add_argument("-v", '--version', help="File version", default='')
    parser.add_argument("-j", '--ncpu', help="Number of cores to use in multiprocessing", default=1, type=int)
    parser.add_argument("-nt", '--n_total_events', help="Total number of events to sample.", default=100, type=int)
    parser.add_argument("-nf", '--n_events_per_file', help="Number of events per file.", default=100, type=int)
    return parser.parse_args()

def handleInput(data):
    if os.path.isfile(data) and ".h5" in os.path.basename(data):
        return [data]
    elif os.path.isfile(data) and ".txt" in os.path.basename(data):
        return sorted([line.strip() for line in open(data,"r")])
    elif os.path.isdir(data):
        return sorted(os.listdir(data))
    elif "*" in data:
        return sorted(glob.glob(data))
    return []

if __name__ == "__main__":
	main()