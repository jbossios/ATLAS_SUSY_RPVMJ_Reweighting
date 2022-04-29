
import h5py
import matplotlib.pyplot as plt
import numpy as np

def main(file_name: str):
  with h5py.File(file_name, 'r') as hf:
    # Get normalization weights
    normweight = np.array(hf['normweight']['values'])
    # Plot number of quark jets
    n_quark_jets = np.array(hf["nQuarkJets"]['values'])
    bins = np.linspace(0, 10, 11)
    plt.hist(n_quark_jets, bins = bins, label='Number of quark-jets', weights = normweight)
    plt.xlabel('Number of quark-jets')
    plt.ylabel('Number of events')
    plt.savefig('n_quark_jets.pdf')
    plt.clf() # clear figure
    # Plot HT distributions
    ht = np.array(hf["HT"]['values'])
    ht_no_quarks = ht[n_quark_jets == 0]
    ht_quarks = ht[n_quark_jets != 0]
    wgt_no_quarks = normweight[n_quark_jets == 0]
    wgt_quarks = normweight[n_quark_jets != 0]
    bins = np.linspace(0, 10000, 101)
    plt.hist(ht_no_quarks, bins = bins, label='No quark-jets', alpha = 0.5 , weights = wgt_no_quarks, density = True)
    plt.hist(ht_quarks, bins = bins, label='At least one quark-jet', alpha = 0.5, weights = wgt_quarks, density = True)
    plt.xlabel('HT [MeV]')
    plt.ylabel('Arbitrary units')
    plt.xscale('log')
    plt.xlim(xmin=500, xmax = 10000)
    plt.legend()
    plt.savefig('HT.pdf')
    print('>>> ALL DONE <<<')

if __name__ == '__main__':
  main('/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/reweighting/Jona/H5_files/v3/mc16a_dijets_JZAll_for_reweighting.h5')
