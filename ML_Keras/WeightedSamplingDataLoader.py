import numpy as np
import h5py
from torch.utils.data import WeightedRandomSampler

# file list
fileList = [
	"user.jbossios.364712.e7142_e5984_s3126_r10724_r10726_p4355.27261089._000001.trees_expanded_spanet.h5",
	"user.jbossios.364708.e7142_e5984_s3126_r10724_r10726_p4355.27261077._000017.trees_expanded_spanet.h5"
]

# first load the normweight
weights = []
fileidx = []
for iF, file in enumerate(fileList):
	with h5py.File(file, "r") as hf:
		normweight = np.array(hf["normweight"]["normweight"]).flatten()
		nevents = normweight.shape[0]
		weights.append(normweight)
		fileidx.append(np.stack([np.full((nevents),iF),np.arange(nevents)],-1))
weights = np.concatenate(weights)
fileidx = np.concatenate(fileidx)
print(weights.shape)
print(fileidx.shape)

# sample event indices
num_batches = 1
batch_size = 100
for i in range(num_batches):
	idx = list(WeightedRandomSampler(weights=weights,num_samples=batch_size,replacement=True))
	
	# get samples and sort by file
	samples = fileidx[idx]
	samples = samples[samples[:, 0].argsort()]
	# print(samples)
	# get unique files and list of events per file
	files = np.unique(samples[:,0])
	events = [samples[np.where(samples[:,0] == i)][:,1] for i in files]
	print(files)
	# print(events)

	x = []
	y = []
	for iF,iE in zip(files,events):
		with h5py.File(fileList[iF], "r") as hf:
			# pick up the kinematics
			m = np.array(hf["source"]["mass"])
			pt = np.array(hf["source"]["pt"])
			eta = np.array(hf["source"]["eta"])
			phi = np.array(hf["source"]["phi"])
			j = np.stack([m,pt,eta,phi],-1)
			x.append(j[iE])
			y.append([])
	
	x = np.concatenate(x)
	y = np.concatenate(y)
	print(x.shape, y.shape)

