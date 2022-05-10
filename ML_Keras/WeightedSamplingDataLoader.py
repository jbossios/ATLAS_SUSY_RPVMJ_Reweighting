import numpy as np
import h5py
import tensorflow as tf
import time

class WeightedSamplingDataLoader(tf.data.Dataset):

    def _generator(probabilities, fileidx, fileList, num_batches, batch_size):
        for iB in range(num_batches):
            start = time.time()
            idx = np.random.choice(range(0,len(probabilities)), size=batch_size, p = probabilities, replace = True)
            print ("Time elapsed for idx:", time.time() - start)
            print(idx)
            start = time.time()
            samples = fileidx[idx]
            samples = samples[samples[:, 0].argsort()]
            # get unique files and list of events per file
            files = np.unique(samples[:,0])
            events = [samples[np.where(samples[:,0] == i)][:,1] for i in files]
            print ("Time elapsed middle:", time.time() - start)
            # create batch
            x = []
            y = []
            start = time.time()
            for iF,iE in zip(files,events):
                with h5py.File(fileList[iF], "r") as hf:
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
            print ("Time elapsed for getting data from files:", time.time() - start)
            yield x, y

    def __new__(self, probabilities, fileidx, fileList, num_batches, batch_size):
        return tf.data.Dataset.from_generator(
            self._generator,
            output_signature = (
                tf.TensorSpec(shape = (None, 8, 4), dtype = tf.float64),
                tf.TensorSpec(shape = (None, ), dtype = tf.float64),
            ),
            args=(probabilities, fileidx, fileList, num_batches, batch_size,)
        )

FileList = "/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/abadea/ParticleNet/v1/Dijets/list.txt"
fileList = sorted([line.strip() for line in open(FileList,"r")])
# fileList = [
#     "/Users/anthonybadea/Documents/ATLAS/rpvmj/ParticleNet/tf-keras/user.jbossios.364708.e7142_e5984_s3126_r10724_r10726_p4355.27261077._000017.trees_expanded_spanet.h5",
#     "/Users/anthonybadea/Documents/ATLAS/rpvmj/ParticleNet/tf-keras/user.jbossios.364712.e7142_e5984_s3126_r10724_r10726_p4355.27261089._000001.trees_expanded_spanet.h5"
# ]
loadWeightSampler = True
if loadWeightSampler:
    x = np.load("WeightSamplerDijets.npz",allow_pickle=True)
    weights = x["weights"]
    fileidx = x["fileidx"]
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
    weights = np.concatenate(weights)
    fileidx = np.concatenate(fileidx)
    np.savez("WeightSamplerDijets.npz",**{"weights":weights,"fileidx":fileidx})


probabilities = weights / weights.sum()

num_batches = 5
batch_size = 1024

dataset = WeightedSamplingDataLoader(probabilities, fileidx, fileList, num_batches, batch_size).prefetch(tf.data.AUTOTUNE)

num_epochs = 1
for epoch_num in range(num_epochs):
        for x,y in dataset:
            print(x.shape,y.shape)

# print(next(WSDataLoader.get_data()))

# sample event indices
# def load_batch():
#   idx = np.random.choice(range(0,len(weights)), size=batch_size, p = weights / weights.sum(), replace = True)
#   samples = fileidx[idx]
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

