from get_data import get_full_data
from make_model import make_model
from torch import as_tensor
from torch import float as tfloat
from torch import load
import numpy as np
import matplotlib.pyplot as plt
from torch import no_grad

def evaluate(model, x, y, normweight):
  x = x.reshape(x.size, 1)
  y = y.reshape(x.size, 1)
  normweight = normweight.reshape(x.size, 1)
  xa = x[y==0]
  xb = x[y==1]
  bins = np.linspace(0, 8000, 100)
  normweightsa = normweight[y==0]
  normweightsb = normweight[y==1]
  with no_grad():
    X = as_tensor(x, dtype=tfloat)
    y = as_tensor(y, dtype=tfloat)
    p = model(X)
  xa = np.multiply(xa, 1000)
  xb = np.multiply(xb, 1000)
  c0, _, _ = plt.hist(xa, bins = bins, alpha = 0.5, weights = normweightsa, label = '#QuarkJets > 0', color = 'red')
  c1, _, _ = plt.hist(xb, bins = bins, alpha = 0.5, weights = normweightsb, label = '#QuarkJets = 0', color = 'blue')
  p = np.array(p).reshape(x.size, 1)
  _pp = p[y==0]
  final_weights = (1-_pp)/_pp
  final_weights *= normweightsa
  c2, _, _ = plt.hist(xa, bins = bins, alpha = 0.5, weights = final_weights, label = '#Quarks > 0 reweighted to #QuarksJets = 0', color = 'yellow') 
  plt.legend()
  plt.show()

if __name__ == '__main__':
  model, loss, opt = make_model(1, 1, lr=1e-3)
  Xs, ys, wgts = get_full_data('/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/reweighting/Jona/H5_files/v1/mc16a_dijets_JZAll_for_reweighting.h5')
  model.load_state_dict(load('model.pt'))
  evaluate(model, Xs, ys, wgts)
