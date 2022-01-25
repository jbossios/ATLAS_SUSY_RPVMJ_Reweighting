import h5py
import os
import sys
import numpy as np

def main(file_name):
  iFile = h5py.File(file_name,"r")
  keys  = list(iFile.keys())
  print('Groups: ')
  print(keys)
  for key in keys:
    print('Subgroups inside {}:'.format(key))
    data      = iFile.get(key) # get group
    subGroups = [x[0] for x in list(data.items())] # get list of subgroups
    print(subGroups)
    for item in subGroups:
      print('Data on {}/{}:'.format(key,item))
      print(np.array(data.get(item)))
      print('size: {}'.format(np.array(data.get(item).size)))

if __name__ == '__main__':
  main('mc16a_dijets_JZAll_for_reweighting.h5')
