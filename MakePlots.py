
import h5py
import matplotlib.pyplot as plt
import numpy as np

def main(file_name: str):
  with h5py.File(file_name, 'r') as hf:
    data = hf.get('data') # get group
    subGroups = [x[0] for x in list(data.items())] # get list of subgroups
    ht = np.array(data.get('HT'))
    normweight = np.array(data.get('normweight'))
    zero_quark_jets_flag = np.array(data.get('ZeroQuarkJetsFlag'))
    ht_no_quarks = ht[zero_quark_jets_flag == 1]
    ht_quarks = ht[zero_quark_jets_flag == 0]
    wgt_no_quarks = normweight[zero_quark_jets_flag == 1]
    wgt_quarks = normweight[zero_quark_jets_flag == 0]
    bins = np.linspace(0, 8000, 100)
    plt.hist(ht_no_quarks, bins = bins, label='No quark-jets', alpha = 0.5 , weights = wgt_no_quarks, density = True)
    plt.hist(ht_quarks, bins = bins, label='At least one quark-jet', alpha = 0.5, weights = wgt_quarks, density = True)
    plt.xlabel('HT [MeV]')
    plt.ylabel('Arbitrary units')
    plt.xscale('log')
    plt.xlim(xmin=500, xmax = 10000)
    plt.legend()
    plt.show()

if __name__ == '__main__':
  main('mc16a_dijets_JZAll_for_reweighting.h5')
