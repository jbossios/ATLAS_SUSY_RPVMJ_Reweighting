
import h5py
import numpy as np
import random

# Idea get full data and divide by number of batches
# Return one array per batch containing X and y

def get_data(file_name: str, batch_size: int = 2048): # FIXME add return type hint
  seed = 10000
  with h5py.File(file_name, 'r') as hf:
    data = hf.get('data') # get group
    # HT
    ht_data = np.array(data.get('HT'))
    random.seed(seed)
    random.shuffle(ht_data)
    n_batches = int(ht_data.size / batch_size)
    ht_data = ht_data[:n_batches*batch_size] # exlude non-full batch
    ht = ht_data.reshape(batch_size, -1)
    # ZeroQuarkJetsFlag
    zero_quark_jets_flag_data = np.array(data.get('ZeroQuarkJetsFlag'))
    random.seed(seed)
    random.shuffle(zero_quark_jets_flag_data)
    zero_quark_jets_flag_data = zero_quark_jets_flag_data[:n_batches*batch_size] # exlude non-full batch
    zero_quark_jets_flag = zero_quark_jets_flag_data.reshape(batch_size, -1)
    # normweight
    normweight_data = np.array(data.get('normweight'))
    random.seed(seed)
    random.shuffle(normweight_data)
    n_batches = int(normweight_data.size / batch_size)
    normweight_data = normweight_data[:n_batches*batch_size] # exlude non-full batch
    normweight = normweight_data.reshape(batch_size, -1)
    return ht, zero_quark_jets_flag, normweight

def get_normweight(file_name: str, batch_size: int = 2048): # FIXME add return type hint
  seed = 10000
  with h5py.File(file_name, 'r') as hf:
    data = hf.get('data') # get group
    normweight_data = np.array(data.get('normweight'))
    random.seed(seed)
    random.shuffle(normweight_data)
    n_batches = int(normweight_data.size / batch_size)
    normweight_data = normweight_data[:n_batches*batch_size] # exlude non-full batch
    normweight = normweight_data.reshape(normweight_data.size, -1)
    return normweight

if __name__ == '__main__':
  X, y = get_data('/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/reweighting/Jona/H5_files/v1/mc16a_dijets_JZAll_for_reweighting.h5')
  print(f'{X = }')
  print(f'{y = }')
