import torch.nn as nn
from torch import save, load
from torch import as_tensor
from torch import float as tfloat
from torch import no_grad
from torch.optim import Adam
from make_model import make_model
from get_data import get_data
import numpy as np
import matplotlib.pyplot as plt
from random import randrange

def train(model: nn.Sequential, loss: nn.BCELoss, opt: Adam, datagen):
  losses = []
  for i, ( Xs, ys) in enumerate(datagen): # datagen was built already with nepochs x batchsize
    # pick up a batch randomly
    X = as_tensor(Xs, dtype=tfloat)
    y = as_tensor(ys, dtype=tfloat)
    p = model(X)
    lossval = loss(p, y)
    
    # Zero the gradients before running the backward pass
    loss.zero_grad()
    
    # Compute gradient of the loss with respect to all the learnable parameters of the model.
    lossval.backward()
    losses.append(lossval.item())
    
    opt.step()
    if i % 100 == 0:
      print(f'lossval = {lossval}')
  return model, losses

if __name__ == '__main__':
  model, loss, opt = make_model(1, 1, lr=1e-3)
  batch_size = 2048
  nepochs = 4000
  datagen = get_data('/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/reweighting/Jona/H5_files/v1/mc16a_dijets_JZAll_for_reweighting.h5', nepochs, batch_size)
  # Train
  model, losses = train(model, loss, opt, datagen)
  plt.plot(losses)
  plt.savefig('losses.pdf')
  save(model.state_dict(), 'model.pt')
