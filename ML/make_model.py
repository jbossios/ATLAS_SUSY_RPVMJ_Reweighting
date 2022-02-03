import torch.nn as nn
from torch.optim import Adam
from typing import Tuple

def make_model(input_dim: int, out_dim: int, lr: float = 1e-4) -> Tuple[nn.Sequential, nn.BCELoss, Adam]:
  model = nn.Sequential(
      nn.Linear(input_dim, 100),
      nn.ReLU(),
      nn.Linear(100, 100),
      nn.ReLU(),
      #nn.Linear(100, 100),
      #nn.ReLU(),
      nn.Linear(100, out_dim),
      nn.Sigmoid()
  )
  print(model)
  
  loss = nn.BCELoss()
 
  opt = Adam(model.parameters(), lr = 1e-4) # lr: learning rate

  return model, loss, opt

if __name__ == '__main__':
  _, _, _ = make_model(1, 1)
