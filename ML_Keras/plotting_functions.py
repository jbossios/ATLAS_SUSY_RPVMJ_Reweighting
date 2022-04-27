'''
Authors: Anthony Badea, Jonathan Bossio
Date: April 27, 2022
'''

import matplotlib.pyplot as plt

def plot_loss(history):
  """
  Plot loss vs epochs
  """
  plt.figure('loss_vs_epoch')
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.yscale('log')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()
  plt.grid(True)
  plt.savefig('loss_vs_epochs.pdf')
