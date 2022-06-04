
import h5py
import numpy as np

# do this per batch
def get_sample(fourmom, nQuarkJets, normweight, probabilities, batch_size, cut_nQuarkJets):
        while True:
                idx = np.random.choice(range(0,len(probabilities)), size=batch_size, p = probabilities, replace = True)
                x = fourmom[idx]
                y = nQuarkJets[idx] >= cut_nQuarkJets # update this cut
                w = normweight[idx]
                yield x, np.stack([y, w],-1)

def main():

        # load this once
        f = h5py.File("user.abadea.364712.e7142_e5984_s3126_r10724_r10726_p4355.29109156._000002.trees_minJetPt50_minNjets6_maxNjets8_v0.h5","r")

        # cuts used
        cut_minAvgMass = 750
        cut_QGTaggerBDT = 0.0 
        cut_nQuarkJets = 2

        # precompute indices
        minAvgMass = np.array(f['EventVars']['minAvgMass'])
        low_minAvgMass = np.where(minAvgMass < cut_minAvgMass)[0] 

        # only keep these loaded in memory
        minAvgMass = minAvgMass[low_minAvgMass]
        normweight = np.array(f['normweight']['normweight'])[low_minAvgMass]
        nQuarkJets = (np.array(f['source']['QGTaggerBDT']) > cut_QGTaggerBDT).sum(1)[low_minAvgMass]
        fourmom = np.stack([f['source']['mass'], f['source']['pt'], f['source']['eta'], f['source']['phi']],-1)[low_minAvgMass]
        fourmom = np.array(f['EventVars']['HT'])[low_minAvgMass]
        probabilities = normweight / normweight.sum()

        # to use four momentum I'll need a better NN like a graphNN to handle the four mom
        train_data_gen = get_sample(fourmom, nQuarkJets, probabilities, 2, cut_nQuarkJets)
        val_data_gen = get_sample(fourmom, nQuarkJets, probabilities, 2, cut_nQuarkJets)

        for i in range(10):
                x,y = next(train_data_gen)
                print(x.shape,y.shape)


if __name__ == "__main__":
        main()
