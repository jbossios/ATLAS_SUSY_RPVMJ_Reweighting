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

	# define regions
	regions = {
		"A" : { "weights" : [], "fileidx" : [], "cuts" : [ ["source/QGTaggerBDT", "le",  [0.5, 2]], ["EventVars/minAvgMass", "le",  0.5] ]},
		"B" : { "weights" : [], "fileidx" : [], "cuts" : [ ["source/QGTaggerBDT", "le",  [0.5, 2]], ["EventVars/minAvgMass", "geq", 0.5] ]},
		"C" : { "weights" : [], "fileidx" : [], "cuts" : [ ["source/QGTaggerBDT", "geq", [0.5, 2]], ["EventVars/minAvgMass", "le",  0.5] ]},
		"D" : { "weights" : [], "fileidx" : [], "cuts" : [ ["source/QGTaggerBDT", "geq", [0.5, 2]], ["EventVars/minAvgMass", "geq", 0.5] ]}
	}

	for iF, file in enumerate(fileList):

	    print(f"File {iF}/{len(fileList)}")

	    with h5py.File(file, "r") as hf:
	    	
	    	# pickup needed variables
	        weights = np.array(hf["normweight/normweight"]).flatten()
	        nevents = weights.shape[0]
	        fileidx = np.stack([np.full((nevents),iF),np.arange(nevents)],-1)

	        # apply cuts and append
	        for key, val in regions.items():
	        	passCuts = np.ones(weights.shape).astype(bool)
	        	for (var, func, cut) in val["cuts"]:
	        		# doCut = np.ones(passCuts.shape)
	        		if "QGTaggerBDT" in var:
	        			temp = doCut(np.array(hf[var]), "geq", cut[0]).sum(1) # cut on BDT score
	        			passCuts = np.logical_and(passCuts, doCut(temp, func, cut[1])) # cut on number of quark tags
        			elif "minAvgMass" in var:
		        		passCuts = np.logical_and(passCuts, doCut(np.array(hf[var]), func, cut)) # cut on minAvgMass value
	        	val["weights"].append(weights[passCuts])
	        	val["fileidx"].append(fileidx[passCuts])

	# prepare job configs
	n_files = int(ops.n_total_events/ops.n_events_per_file)
	configs = []
	for key, val in regions.items():

		# concat and save
		val["weights"] = np.concatenate(val["weights"])
		val["fileidx"] = np.concatenate(val["fileidx"])
		np.savez(f"{ops.sample}_Region{key}_SamplingWeights.npz",**{"weights":weights,"fileidx":fileidx})

		# don't sample from null regions
		if all(val["weights"] == 0):
			print(f"Region {key} has all zero weights")
			continue

		# compute probabilities
		val["probabilities"] = val["weights"] / (val["weights"].sum() + 10**-50)

		# make conf
		for iF in range(n_files):
			configs.append({
				"probabilities" : val["probabilities"], # this will make it so the same event can be used more than once
				"fileidx" : val["fileidx"],
				"fileList" : fileList,
				"n_samples" : ops.n_events_per_file,
				"outName" : os.path.join(ops.outDir, f"{ops.sample}_Region{key}_v{ops.version}_{iF}.npz")
			})
	
	# launch sampling
	if ops.ncpu > 1:
		results = Pool(ncpu).map(add_normweight, configs)
	else:
		for conf in configs:
			print(f"Creating {conf['outName']}")
			sample(conf)

def doCut(x, func, cut):
	if func == "le":
		return x < cut
	elif func == "leq":
		return x <= cut
	elif func == "ge":
		return x > cut
	elif func == "geq":
		return x >= cut
	return np.ones(x.shape)

def sample(conf):
    # get sample list
    idx = np.random.choice(range(0,len(conf["probabilities"])), size=conf["n_samples"], p = conf["probabilities"], replace = False)
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
    parser.add_argument('-s', '--sample',  help="Sample name", default='Dijets')
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
        return sorted([os.path.join(data,i) for i in os.listdir(data)])
    elif "*" in data:
        return sorted(glob.glob(data))
    return []

if __name__ == "__main__":
	main()
