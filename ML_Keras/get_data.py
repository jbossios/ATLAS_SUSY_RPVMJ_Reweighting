import h5py
import numpy as np
import random
from sklearn.model_selection import train_test_split

# Need the following to run on LXPLUS
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from typing import Tuple
import time

def get_data(file_name: str, nepochs: int, batch_size: int = 2048, seed: int = None, debug: bool = False): # FIXME add return type hint
  """
  Sample nepochs batches of size batch_size
  On each batch, data is sampled from pdfs constructed using weighted distributions
  data is sampled from the corresponding pdf based on probabilities
  *** Returns a generator ***
  """
  scale = 1000 #scale down HT to values closer to unity
  fudgefactor = 1 #if >1, artificially make the separation between both distributions better
  while True:
    with h5py.File(file_name, 'r') as hf:
      # Get bin probabilities for ht distributions for events w/ and w/o quark jets
      nQuarkJets = np.array(hf["nQuarkJets"]['values'])
      # Get HT for events w/ and w/o quark jets
      ht = np.array(hf['HT']['values'])/scale
      ht_zq = ht[nQuarkJets == 0]
      ht_nzq = ht[nQuarkJets > 0]*fudgefactor
      # Get normalization weight (normweight) for events w/ and w/o quark jets
      wgt = np.array(hf['normweight']['values'])
      wgt_zq = wgt[nQuarkJets == 0]
      wgt_nzq = wgt[nQuarkJets > 0]
      # Define binning for HT data
      bin_width = 80/scale
      min_bin = 0
      max_bin = 8000/scale
      n_bins = int((max_bin - min_bin)/bin_width)
      bins = np.linspace(min_bin, max_bin, n_bins + 1)
      bin_centers = np.linspace(0.5*bin_width, max_bin+0.5*bin_width , n_bins)
      if debug: # plot input HT histograms
        c0, _, _ = plt.hist(ht_zq, bins = bins, weights = wgt_zq, alpha = 0.5, color = 'red', density = True)
        c1, _, _ = plt.hist(ht_nzq, bins = bins, weights = wgt_nzq, alpha = 0.5, color = 'blue', density = True)
        plt.savefig('compare_input_HT_histograms.pdf')  # TODO: improve output name
      # Construct pdfs
      p_zq, _ = np.histogram(ht_zq, bins = bins, weights = wgt_zq, density = True) # pdf for HT distribution on events w/ quark jets
      p_nzq, _ = np.histogram(ht_nzq, bins = bins, weights = wgt_nzq, density = True) # pdf for HT distribution on events w/o quark jets
      # p_flag, _ = np.histogram(quark_jet_flag, bins = np.linspace(-0.5, 1.5, 3), weights = wgt, density = True) # pdf to decide if event has or not quark jets
      # Prepare batches of data
      for iepoch in range(nepochs): # loop over batches
        if debug:
          print(f'iepoch = {iepoch}')
        # Decide how many events will have quark jets and how many will not
        flag_sample = np.random.choice(np.linspace(0, 1, 2), batch_size)
        zq_size = np.count_nonzero(flag_sample == 1)
        nzq_size = np.count_nonzero(flag_sample == 0)
        if debug:
          print(f'zq_size = {zq_size}')
          print(f'nzq_size = {nzq_size}')
        # Sample the corresponding number of HT values from the appropriate pdf
        ht_zq_sample = np.random.choice(ht_zq, zq_size, p=wgt_zq/wgt_zq.sum())
        zq_flags = np.tile([1], zq_size)
        ht_nzq_sample = np.random.choice(ht_nzq, nzq_size, p=wgt_nzq/wgt_nzq.sum())
        nzq_flags = np.tile([0], nzq_size)
        # Concatenate HT values from both type of events (w/ and w/o quark jets)
        ht_sample = np.concatenate((ht_zq_sample, ht_nzq_sample), axis = 0)
        flags_sample = np.concatenate((zq_flags, nzq_flags), axis = 0)
        # Reshape data and shuffle coherently
        ht_sample_shaped = ht_sample.reshape(batch_size, -1)
        flags_sample_shaped = flags_sample.reshape(batch_size, -1)
        if seed is not None:
          random.seed(seed)
        random.shuffle(ht_sample_shaped)
        if seed is not None:
          random.seed(seed)
        random.shuffle(flags_sample_shaped)
        plt.clf() # clean figure
        if not iepoch and debug: # compare sampled data for first epoch
          c2, _, _ = plt.hist(ht_zq_sample, bins = bins, alpha = 0.5, color = 'green', density = True)
          c3, _, _ = plt.hist(ht_nzq_sample, bins = bins, alpha = 0.5, color = 'orange', density = True)
          plt.savefig('compare_sampled_data.pdf')  # TODO: improve output name
        yield np.array(ht_sample_shaped), np.array(flags_sample_shaped)

def get_data_ABCD(file_name: str, nepochs: int, batch_size: int = 10000, seed: int = None, train: bool = True, test_sample: bool = False):
  """
  Sample nepochs batches of size batch_size
  On each batch, data is sampled from pdfs constructed using weighted distributions
  data is sampled from the corresponding pdf based on probabilities
  *** Returns a generator ***
  """
  # cuts used
  cut_minAvgMass = 750
  cut_QGTaggerBDT = 0.14
  cut_nQuarkJets = 2
  with h5py.File(file_name, 'r') as f:
    # pick up variables from file
    x = np.stack([
                np.array(f['EventVars']['HT']),
                np.array(f['EventVars']['deta']),
                np.array(f['EventVars']['djmass']),
                np.array(f['EventVars']['minAvgMass']),
                np.array(f['source']['pt'][:,0])
           ],-1)
    minAvgMass = np.array(f['EventVars']['minAvgMass'])
    nQuarkJets = (np.array(f['source']['QGTaggerBDT']) > cut_QGTaggerBDT).sum(1)
    normweight = np.array(f['normweight']['normweight'])
    # normweight = np.sqrt(normweight)
    nEvents = minAvgMass.shape[0]
    #print(f"Number of events: {nEvents}")
    """  
    For each batch, learn 0->1 or 1->2.
    For 0->1, train without cuts. For 1->2, we need the minavg cut.
    Whether we learn 0->1 or 1->2 is alternate for each batch.
    """
    # Reweight 1->2. Create cuts to reweight A -> C
    RegA_12 = np.logical_and(minAvgMass < cut_minAvgMass, nQuarkJets == 1)
    RegC_12 = np.logical_and(minAvgMass < cut_minAvgMass, nQuarkJets >= cut_nQuarkJets)
    # Reweight 0->1. 
    RegA_01 = nQuarkJets == 0
    RegC_01 = nQuarkJets == 1
    # Evaluate B->D.
    RegB = np.logical_and(minAvgMass >= cut_minAvgMass, nQuarkJets == 1)
    RegD = np.logical_and(minAvgMass >= cut_minAvgMass, nQuarkJets >= cut_nQuarkJets)
    
    def get_events(RegA, RegC):
      # get events per region
      RegA_x = x[RegA]
      RegA_weights = normweight[RegA]
      RegA_y = np.zeros(RegA_weights.shape)

      RegC_x = x[RegC]
      RegC_weights = normweight[RegC]
      RegC_y = np.ones(RegC_weights.shape)

      RegB_x = x[RegB]
      RegD_x = x[RegD]

      # return the full sample if not train       
      if not train:
        print('Full sample for evaluation:')
        # normalize for prediction
        RegA_x = (RegA_x-np.mean(RegA_x,0))/np.std(RegA_x,0)
        RegC_x = (RegC_x-np.mean(RegC_x,0))/np.std(RegC_x,0)
        RegB_x = (RegB_x-np.mean(RegB_x,0))/np.std(RegB_x,0)
        RegD_x = (RegD_x-np.mean(RegD_x,0))/np.std(RegD_x,0)
        return RegA_x, RegB_x, RegD_x, RegC_x
      # combine with same statistics
      nEventsA = -1 #min(RegA_y.shape[0],RegC_y.shape[0])
      nEventsC = -1 #2*nEvents
      X = np.concatenate([RegA_x[:nEventsA],RegC_x[:nEventsC]])
      Y = np.concatenate([RegA_y[:nEventsA],RegC_y[:nEventsC]])
      W = np.concatenate([RegA_weights[:nEventsA],RegC_weights[:nEventsC]])
      Y = np.stack([Y,W],axis=-1)
      # standardize
      X = (X - np.mean(X,0))/np.std(X,0)
      #print(f"X mean, std: {np.mean(X)}, {np.std(X)}")
      return X, Y

    X_12, Y_12 = get_events(RegA_12, RegC_12)
    X_01, Y_01 = get_events(RegA_01, RegC_01)

    X_12_train, X_12_test, Y_12_train, Y_12_test = train_test_split(X_12, Y_12, test_size=0.25, shuffle=True)
    X_01_train, X_01_test, Y_01_train, Y_01_test = train_test_split(X_01, Y_01, test_size=0.25, shuffle=True)

    while True:
      # record time
      start_time = time.time()
      # Prepare batches of data
      nbatch = int(nEvents/batch_size)
      next_sample_source = 0
      for ibatch in range(nbatch): 
        if next_sample_source == 0:
          X_train, X_test, Y_train, Y_test = X_01_train, X_01_test, Y_01_train, Y_01_test
          next_sample_source = 1
        else:
          X_train, X_test, Y_train, Y_test = X_12_train, X_12_test, Y_12_train, Y_12_test
          next_sample_source = 0
        start = 0
        if not test_sample:
          X_train[start:start+batch_size]
          Y_train[start:start+batch_size]
          start = start+batch_size
          yield X_train, Y_train
        else:
          yield X_test, Y_test
      print("--- %s seconds ---" % (time.time() - start_time))


def get_full_data(file_name: str) -> Tuple[np.array, np.array, np.array]:
  """
  Get full (actual) data from input H5 file (not sampling from pdfs!)
  """
  scale = 1000 #scale down HT to values closer to unity
  fudgefactor = 1 #if >1, artificially make the separation between both distributions better
  with h5py.File(file_name, 'r') as hf:
    # Get bin probabilities for ht distributions for events w/ and w/o quark jets
    nQuarkJets = np.array(hf["nQuarkJets"]['values'])
    # Get HT for events w/ and w/o quark jets
    ht = np.array(hf['HT']['values'])/scale
    # Get normalization weight (normweight) for events w/ and w/o quark jets
    wgt = np.array(hf['normweight']['values'])
    return ht, nQuarkJets, wgt

if __name__ == '__main__':
  # X, y = next(get_data('/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/reweighting/Jona/H5_files/v2/mc16a_dijets_JZAll_for_reweighting.h5', 1000, 100000, None, True))
  # print(f'X[0] = {X[0]}')
  # print(f'y[0] = {y[0]}')
  train_data_gen = get_data_ABCD('../../input_file/user.abadea.DijetsALL_minJetPt50_minNjets6_maxNjets8_v1.h5', 5 , 10000, train=True, test_sample=False)
  X, y = next(train_data_gen)
  print(f'X[0] = {X[0]}')
  print(f'y[0] = {y[0]}')
  # X, y, wgt = get_full_data('/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/reweighting/Jona/H5_files/v1/mc16a_dijets_JZAll_for_reweighting.h5')
  # print(f'X[0] = {X[0]}')
  # print(f'y[0] = {y[0]}')
  # print(f'wgt[0] = {wgt[0]}')
