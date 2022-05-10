import numpy as np
import h5py
import tensorflow as tf

# file list
# fileList = "/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/abadea/ParticleNet/v1/Dijets/list.txt"
# fileList = sorted([line.strip() for line in open(fileList,"r")])

# # first load the normweight
# loadWeightSampler = True

# if loadWeightSampler:
#         x = np.load("WeightSamplerDijets.npz",allow_pickle=True)
#         weights = x["weights"]
#         fileidx = x["fileidx"]
# else:
#         weights = []
#         fileidx = []
#         for iF, file in enumerate(fileList):
#                 print(f"File {iF}/{len(fileList)}")
#                 with h5py.File(file, "r") as hf:
#                         normweight = np.array(hf["normweight"]["normweight"]).flatten()
#                         nevents = normweight.shape[0]
#                         weights.append(normweight)
#                         fileidx.append(np.stack([np.full((nevents),iF),np.arange(nevents)],-1))
#         weights = np.concatenate(weights)
#         fileidx = np.concatenate(fileidx)
#         np.savez("WeightSamplerDijets.npz",**{"weights":weights,"fileidx":fileidx})

# print(weights.shape)
# print(fileidx.shape)



class WeightedSamplingDataLoader():
	
	def __init__(self, WeightSampler, FileList, num_batches, batch_size):
		# load sampled weights
		if WeightSampler.endswith(".npz"):
			x = np.load(WeightSampler,allow_pickle=True)
	        self.weights = x["weights"]
	        self.p = self.weights / self.weights.sum()
        else:
       		weights = []
	        fileidx = []
	        for iF, file in enumerate(fileList):
	                print(f"File {iF}/{len(fileList)}")
	                with h5py.File(file, "r") as hf:
	                        normweight = np.array(hf["normweight"]["normweight"]).flatten()
	                        nevents = normweight.shape[0]
	                        weights.append(normweight)
	                        fileidx.append(np.stack([np.full((nevents),iF),np.arange(nevents)],-1))
	        self.weights = np.concatenate(weights)
	        self.fileidx = np.concatenate(fileidx)
	        # np.savez("WeightSamplerDijets.npz",**{"weights":weights,"fileidx":fileidx})
        self.fileidx = x["fileidx"]
		self.num_batches = num_batches
		self.batch_size = batch_size
		self.fileList = sorted([line.strip() for line in open(FileList,"r")])

    def get_data(self):
    	with True:
    		for iB in range(num_batches):
    			idx = np.random.choice(range(0,len(self.weights)), size=self.batch_size, p = self.p, replace = True)
    			samples = self.fileidx[idx]
			    samples = samples[samples[:, 0].argsort()]
			    # get unique files and list of events per file
			    files = np.unique(samples[:,0])
			    events = [samples[np.where(samples[:,0] == i)][:,1] for i in files]
			    # create batch
			    x = []
			    y = []
			    for iF,iE in zip(files,events):
			            with h5py.File(self.fileList[iF], "r") as hf:
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
			    yield x,y

WeightSampler = "WeightSamplerDijets.npz"
FileList = "/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/abadea/ParticleNet/v1/Dijets/list.txt"
num_batches = 5
batch_size = 1024
WSDataLoader = WeightedSamplingDataLoader(WeightSampler, FileList, num_batches, batch_size)
print(next(WSDataLoader.get_data()))

# sample event indices
# def load_batch():
# 	idx = np.random.choice(range(0,len(weights)), size=batch_size, p = weights / weights.sum(), replace = True)
# 	samples = fileidx[idx]
#     samples = samples[samples[:, 0].argsort()]
#     # get unique files and list of events per file
#     files = np.unique(samples[:,0])
#     events = [samples[np.where(samples[:,0] == i)][:,1] for i in files]
#     # create batch
#     x = []
#     y = []
#     for iF,iE in zip(files,events):
#             with h5py.File(fileList[iF], "r") as hf:
#                     # pick up the kinematics
#                     m = np.array(hf["source"]["mass"])
#                     pt = np.array(hf["source"]["pt"])
#                     eta = np.array(hf["source"]["eta"])
#                     phi = np.array(hf["source"]["phi"])
#                     j = np.stack([m,pt,eta,phi],-1)
#                     x.append(j[iE])
#                     y.append([])       
#     x = np.concatenate(x)
#     y = np.concatenate(y)
#     print(x.shape, y.shape)
#     return x,y



# for i in range(num_batches):
#         #idx = list(WeightedRandomSampler(weights=weights,num_samples=batch_size,replacement=True))
#         idx = np.random.choice(range(0,len(weights)), size=batch_size, p = weights / weights.sum(), replace = True)
#         # get samples and sort by file
#         samples = fileidx[idx]
#         samples = samples[samples[:, 0].argsort()]
#         # print(samples)
#         # get unique files and list of events per file
#         files = np.unique(samples[:,0])
#         events = [samples[np.where(samples[:,0] == i)][:,1] for i in files]
#         print(files)
        # print(events)

        # x = []
        # y = []
        # for iF,iE in zip(files,events):
        #         with h5py.File(fileList[iF], "r") as hf:
        #                 # pick up the kinematics
        #                 m = np.array(hf["source"]["mass"])
        #                 pt = np.array(hf["source"]["pt"])
        #                 eta = np.array(hf["source"]["eta"])
        #                 phi = np.array(hf["source"]["phi"])
        #                 j = np.stack([m,pt,eta,phi],-1)
        #                 x.append(j[iE])
        #                 y.append([])
                        
        # x = np.concatenate(x)
        # y = np.concatenate(y)
        # print(x.shape, y.shape)

