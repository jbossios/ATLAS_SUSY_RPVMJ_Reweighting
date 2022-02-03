#from torch import *
import torch.nn as nn
from torch import save, load
from torch import Tensor
from torch import no_grad
from torch.optim import Adam
from make_model import make_model
from get_data import get_data, get_normweight
import numpy as np
import matplotlib.pyplot as plt
from random import randrange

# TODO
# Take into account normweight
# Move evaluate to another script

def train(model: nn.Sequential, loss: nn.BCELoss, opt: Adam, Xs: np.ndarray, ys: np.ndarray, n_epochs: int = 500):
  for i in range(n_epochs): # loop over epochs
    # pick up a batch randomly
    item = randrange(Xs.size/batch_size)
    X = Xs[item].reshape(Xs[item].size, 1)
    y = ys[item].reshape(Xs[item].size, 1)
    p = model(Tensor(X).float())
    lossval = loss(p, Tensor(y).float())
    
    # Zero the gradients before running the backward pass
    loss.zero_grad()
    
    # Compute gradient of the loss with respect to all the learnable parameters of the model.
    lossval.backward()
    
    opt.step()
    if i % 100 == 0:
      print(f'{lossval = }')
  return model

# Temporary (move it elsewhere, but need to save model when training)
def evaluate(model, x, y, normweight):
  x = np.array(x).reshape(x.size, 1)
  y = np.array(y).reshape(x.size, 1)
  xa = x[y==0]
  xb = x[y==1]
  bins = np.linspace(0, 8000, 100)
  normweightsa = normweight[y==0]
  normweightsb = normweight[y==1]
  c0, _, _ = plt.hist(xa, bins = bins, alpha = 0.5, weights = normweightsa, label = '#QuarkJets > 0', color = 'red', density = True)
  c1, _, _ = plt.hist(xb, bins = bins, alpha = 0.5, weights = normweightsb, label = '#QuarkJets = 0', color = 'blue', density = True)
  #c0, _, _ = plt.hist(xa, bins = bins, alpha = 0.5, label = '#QuarkJets > 0', color = 'red')
  #c1, _, _ = plt.hist(xb, bins = bins, alpha = 0.5, label = '#QuarkJets = 0', color = 'blue')
  with no_grad():
    #p = model(Tensor(x))[:,0]
    p = model(Tensor(x).float())
  p = np.array(p)
  y = y.reshape(1, len(y))[0]
  print(f'{p = }')
  _pp = p[y==1]
  print(f'{_pp = }')
  #c2, _, _ = plt.hist(xb, bins = bins, alpha = 0.5, weights = normweightsb*(1-_pp)/_pp, label = '#Quarks > 0 reweighted to #QuarksJets = 0', color = 'yellow') 
  c2, _, _ = plt.hist(xb, bins = bins, alpha = 0.5, weights = (1-_pp)/_pp, label = '#Quarks > 0 reweighted to #QuarksJets = 0', color = 'yellow') 
  plt.legend()
  plt.show()

if __name__ == '__main__':
  model, loss, opt = make_model(1, 1)
  #batch_size = 2048
  batch_size = 4096
  Xs, ys, _ = get_data('/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/reweighting/Jona/H5_files/v1/mc16a_dijets_JZAll_for_reweighting.h5', batch_size)
  # Train
  model = train(model, loss, opt, Xs, ys, 10000) # Temporary
  #model = train(model, loss, opt, Xs, ys, wgts, int(Xs.size/batch_size)) # Temporary
  #model = train(model, loss, opt, Xs, ys, int(Xs.size/batch_size)) # Temporary
  #model = train(model, loss, opt, Xs, ys, 10) # Temporary
  save(model.state_dict(), 'model.pt')
  # Evaluate
  #normweight = get_normweight('/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/reweighting/Jona/H5_files/v1/mc16a_dijets_JZAll_for_reweighting.h5', batch_size)
  #model.load_state_dict(load('model.pt'))
  #evaluate(model, Xs, ys, normweight)
