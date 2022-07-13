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

def get_data_ABCD(file_name: str, nepochs: int, batch_size: int = 10000, seed: int = None, test_sample: str = None):
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
    RegB_x = x[RegB]
    RegD_x = x[RegD]
    # get events for regions.
    def get_events(RegA, RegC, RegCat):
      RegA_weights = normweight[RegA]
      RegA_y = np.zeros(RegA_weights.shape)
      RegC_weights = normweight[RegC]
      RegC_y = np.ones(RegC_weights.shape)
      if RegCat == 1:
        RegA_x = np.concatenate((x[RegA],np.ones([len(RegA_y),1], dtype=int)), axis =1)
        RegC_x = np.concatenate((x[RegC],np.ones([len(RegC_y),1], dtype=int)), axis =1)
      elif RegCat == 0:
        RegA_x = np.concatenate((x[RegA],np.zeros([len(RegA_y),1], dtype=int)), axis =1)
        RegC_x = np.concatenate((x[RegC],np.zeros([len(RegC_y),1], dtype=int)), axis =1)
      elif RegCat == 0:
        print('Invalid RegCat.')
        RegA_x = x[RegA]
        RegC_x = x[RegC]
      # combine with same statistics
      nEventsA = -1 #min(RegA_y.shape[0],RegC_y.shape[0])
      nEventsC = -1 #2*nEvents
      X = np.concatenate([RegA_x[:nEventsA],RegC_x[:nEventsC]])
      Y = np.concatenate([RegA_y[:nEventsA],RegC_y[:nEventsC]])
      W = np.concatenate([RegA_weights[:nEventsA],RegC_weights[:nEventsC]])
      Y = np.stack([Y,W],axis=-1)
      # standardize
      #X = (X - np.mean(X,0))/np.std(X,0)
      #print(f"X mean, std: {np.mean(X)}, {np.std(X)}")
      return X, Y
    # Get events
    X_12, Y_12 = get_events(RegA_12, RegC_12, 1)
    X_01, Y_01 = get_events(RegA_01, RegC_01, 0)
    X_12_train, X_12_test, Y_12_train, Y_12_test = train_test_split(X_12, Y_12, test_size=0.25, shuffle=True)
    X_01_train, X_01_test, Y_01_train, Y_01_test = train_test_split(X_01, Y_01, test_size=0.25, shuffle=True)
    # Load data
    while True:
      # record time
      start_time = time.time()
      # Prepare batches of data
      nbatch = int(nEvents/batch_size)*2
      next_sample_source = 0
      start = 0
      for ibatch in range(nbatch): 
        # Generate training samples
        if test_sample==None:
          # check sample source
          if next_sample_source == 0:
            X_train = X_01_train[start:start+batch_size]
            Y_train = Y_01_train[start:start+batch_size]
            next_sample_source = 1
          else:
            X_train = X_12_train[start:start+batch_size]
            Y_train = Y_12_train[start:start+batch_size]
            start = start+batch_size
            next_sample_source = 0
          yield X_train, Y_train
        # Generate validation samples
        elif test_sample == "01": 
          yield X_01_test, Y_01_test
        elif test_sample == "12": 
          yield X_12_test, Y_12_test
        elif test_sample == "012":  
          yield np.concatenate((X_01_test, X_12_test),axis=0), np.concatenate((Y_01_test, Y_12_test),axis=0)
        elif test_sample == "BD": 
          yield get_events(RegB, RegD)
        else:
          print("Invalid test_sample: ", test_sample, ". Return test_sample = 012.")
      # Record time for all batches.
      print("--- %s seconds ---" % (time.time() - start_time))

def get_full_data_ABCD(file_name: str) -> Tuple[np.array, np.array, np.array, np.array]:
  # cuts used
  cut_minAvgMass = 750
  cut_QGTaggerBDT = 0.14
  cut_nQuarkJets = 2
  with h5py.File(file_name, 'r') as f:
    # pick up variables from file
    minAvgMass = np.array(f['EventVars']['minAvgMass'])
    nQuarkJets = (np.array(f['source']['QGTaggerBDT']) > cut_QGTaggerBDT).sum(1)
    nEvents = minAvgMass.shape[0]
    x = np.stack([
                np.array(f['EventVars']['HT']),
                np.array(f['EventVars']['deta']),
                np.array(f['EventVars']['djmass']),
                np.array(f['EventVars']['minAvgMass']),
                np.array(f['source']['pt'][:,0]),
                np.ones(nEvents, dtype=int)
           ],-1)
    RegA_x = x[np.logical_and(minAvgMass < cut_minAvgMass, nQuarkJets == 1)]
    RegC_x = x[np.logical_and(minAvgMass < cut_minAvgMass, nQuarkJets >= cut_nQuarkJets)]
    RegB_x = x[np.logical_and(minAvgMass >= cut_minAvgMass, nQuarkJets == 1)]
    RegD_x = x[np.logical_and(minAvgMass >= cut_minAvgMass, nQuarkJets >= cut_nQuarkJets)]
    # normalize for prediction
    # RegA_x = (RegA_x-np.mean(RegA_x,0))/np.std(RegA_x,0)
    # RegC_x = (RegC_x-np.mean(RegC_x,0))/np.std(RegC_x,0)
    # RegB_x = (RegB_x-np.mean(RegB_x,0))/np.std(RegB_x,0)
    # RegD_x = (RegD_x-np.mean(RegD_x,0))/np.std(RegD_x,0)
    return RegA_x, RegB_x, RegC_x, RegD_x

def get_full_weights_ABCD(file_name: str) -> Tuple[np.array, np.array, np.array, np.array]:
  # cuts used
  cut_minAvgMass = 750
  cut_QGTaggerBDT = 0.14
  cut_nQuarkJets = 2
  with h5py.File(file_name, 'r') as f:
    # pick up variables from file
    minAvgMass = np.array(f['EventVars']['minAvgMass'])
    nQuarkJets = (np.array(f['source']['QGTaggerBDT']) > cut_QGTaggerBDT).sum(1)
    normweight = np.array(f['normweight']['normweight'])
    RegA_weights = normweight[np.logical_and(minAvgMass < cut_minAvgMass, nQuarkJets == 1)]
    RegC_weights = normweight[np.logical_and(minAvgMass < cut_minAvgMass, nQuarkJets >= cut_nQuarkJets)]
    RegB_weights = normweight[np.logical_and(minAvgMass >= cut_minAvgMass, nQuarkJets == 1)]
    RegD_weights = normweight[np.logical_and(minAvgMass >= cut_minAvgMass, nQuarkJets >= cut_nQuarkJets)]
    return RegA_weights, RegB_weights, RegC_weights, RegD_weights

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
  train_data_gen = get_data_ABCD('../../input_file/user.abadea.DijetsALL_minJetPt50_minNjets6_maxNjets8_v1.h5', 3 , 10000, test_sample=None)
  test_data_gen = get_data_ABCD('../../input_file/user.abadea.DijetsALL_minJetPt50_minNjets6_maxNjets8_v1.h5', 3 , 10000, test_sample='012')

  i = 0
  for ibatch in train_data_gen:
    print("train_data_gen ", i)
    x, y = ibatch
    print('x: ', x.shape)
    print('y: ', y.shape)
    i += 1
    if i > 5:
      break
  j = 0
  for jbatch in test_data_gen:
    print("test_data_gen ", j)
    x, y = jbatch
    print('x: ', x.shape)
    print('y: ', y.shape)
    j += 1
    if j > 5:
      break

  RegA_x, RegB_x, RegC_x, RegD_x = get_full_data_ABCD('../../input_file/user.abadea.DijetsALL_minJetPt50_minNjets6_maxNjets8_v1.h5')
  print('A: ', RegA_x.shape)
  print('B: ', RegB_x.shape)
  print('C: ', RegC_x.shape)
  print('D: ', RegD_x.shape)

  # X, y, wgt = get_full_data('/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/reweighting/Jona/H5_files/v1/mc16a_dijets_JZAll_for_reweighting.h5')
  # print(f'X[0] = {X[0]}')
  # print(f'y[0] = {y[0]}')
  # print(f'wgt[0] = {wgt[0]}')
