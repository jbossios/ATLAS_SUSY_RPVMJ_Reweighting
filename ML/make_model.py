import torch.nn as nn
from torch.optim import Adam
from typing import Tuple

def make_model(input_dim: int, out_dim: int, lr: float = 1e-4) -> Tuple[nn.Sequential, nn.BCELoss, Adam]:
  block = 10
  model = nn.Sequential(
      #nn.Linear(input_dim, block),
      #nn.ReLU(),
      #nn.Linear(block, block),
      #nn.ReLU(),
      #nn.Linear(block, block),
      #nn.ReLU(),
      #nn.Linear(block, out_dim),
      #nn.Sigmoid()

      nn.Linear(input_dim, out_dim),
      nn.Sigmoid()
  )
  print(model)
  
  loss = nn.BCELoss()
 
  opt = Adam(model.parameters(), lr = lr) # lr: learning rate

  return model, loss, opt

if __name__ == '__main__':
  _, _, _ = make_model(1, 1)
