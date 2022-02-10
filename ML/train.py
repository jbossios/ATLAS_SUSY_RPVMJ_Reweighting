import torch.nn as nn
from torch import save, load
from torch import Tensor
from torch import no_grad
from torch import randn # Temporary
from torch.optim import Adam
from make_model import make_model
#from get_data import get_data, get_normweight
from get_data import get_data
import numpy as np
import matplotlib.pyplot as plt
from random import randrange

def train(model: nn.Sequential, loss: nn.BCELoss, opt: Adam, Xs: np.ndarray, ys: np.ndarray, n_epochs: int = 500):
  for i in range(n_epochs): # loop over epochs
    # pick up a batch randomly
    #item = randrange(Xs.size/batch_size)
    X = Xs[i].reshape(Xs[i].size, 1)
    y = ys[i].reshape(Xs[i].size, 1)
    p = model(Tensor(X).float())

    lossval = loss(p, Tensor(y).float())
    
    # Zero the gradients before running the backward pass
    loss.zero_grad()
    
    # Compute gradient of the loss with respect to all the learnable parameters of the model.
    lossval.backward()
    
    opt.step()
    if i % 100 == 0:
      print(f'lossval = {lossval}')
  return model

if __name__ == '__main__':
  model, loss, opt = make_model(1, 1)
  batch_size = 2048
  nepochs = 10000
  #Xs, ys, _ = get_data('/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/reweighting/Jona/H5_files/v1/mc16a_dijets_JZAll_for_reweighting.h5', nepochs, batch_size)
  Xs, ys = get_data('/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/reweighting/Jona/H5_files/v1/mc16a_dijets_JZAll_for_reweighting.h5', nepochs, batch_size)
  # Train
  model = train(model, loss, opt, Xs, ys, nepochs) # Temporary
  save(model.state_dict(), 'model.pt')
