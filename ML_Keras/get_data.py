import h5py
import numpy as np
import random

# Need the following to run on LXPLUS
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from typing import Tuple

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
      ht_zq = ht[nQuarkJets > 0]
      ht_nzq = ht[nQuarkJets == 0]*fudgefactor
      # Get normalization weight (normweight) for events w/ and w/o quark jets
      wgt = np.array(hf['normweight']['values'])
      wgt_zq = wgt[nQuarkJets > 0]
      wgt_nzq = wgt[nQuarkJets == 0]
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
  X, y = next(get_data('/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/reweighting/Jona/H5_files/v1/mc16a_dijets_JZAll_for_reweighting.h5', 1000, 100000, None, True))
  print(f'X[0] = {X[0]}')
  print(f'y[0] = {y[0]}')
  # X, y, wgt = get_full_data('/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/reweighting/Jona/H5_files/v1/mc16a_dijets_JZAll_for_reweighting.h5')
  # print(f'X[0] = {X[0]}')
  # print(f'y[0] = {y[0]}')
  # print(f'wgt[0] = {wgt[0]}')
