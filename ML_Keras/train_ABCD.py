'''
Authors: Anthony Badea, Jonathan Bossio
Date: Monday April 25, 2022
'''

# python imports
import h5py
import tensorflow as tf
import tensorflow.keras.backend as K
import argparse
import json
import datetime
import os
import sys
import logging
import pandas as pd
import random as python_random
import numpy as np
from sklearn.model_selection import train_test_split
import gc

# custom imports
try:
    from get_data import get_data_ABCD
    from make_model import simple_model, simple_model_norm, sqrtR_loss, mean_pred
except:
    from ML_Keras.get_data import get_data_ABCD
    from ML_Keras.make_model import simple_model, simple_model_norm, sqrtR_loss, mean_pred

# Tensorflow GPU settings
physical_devices = tf.config.list_physical_devices('GPU') 
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# This function keeps the initial learning rate for the first N epochs and decreases it exponentially after that.
def scheduler(epoch, lr):
  if epoch < 5:
    return lr
  else:
    return lr * tf.math.exp(-0.1)

def main(config = None):

    # logger
    logging.basicConfig(level = 'INFO', format = '%(levelname)s: %(message)s')
    log = logging.getLogger()

    # user options
    ops = options()
    if config is not None:
        conf = config
    else:
        if ops.conf:
            with open(ops.conf) as f:
                log.info(f"opening {ops.conf}")
                conf = json.load(f)
        else:
            conf = {
                "file": ops.inFile,
                "nepochs": ops.nepochs,
                "batch_size": ops.batch_size,
                "learning_rate" : ops.learning_rate,
                "input_dim" : ops.input_dim,
                "seed" : ops.seed
            }
            with open('conf.json', 'w') as fp:
                json.dump(conf, fp)

    # protection
    if not conf["file"]:
        log.error('ERROR: No input file was provided, exiting')
        sys.exit(1)

    # training configuration
    log.info("Training configuration: \n" + json.dumps(conf, indent=4, sort_keys=True))

    # data set generators
    seed = None
    if "seed" in conf and conf["seed"] is not None:
        seed = conf["seed"]

    train_data_gen = get_data_ABCD(file_name=conf["file"], nepochs=conf["nepochs"],batch_size=conf["batch_size"], seed=seed, test_sample=None)
    val_data_gen = get_data_ABCD(file_name=conf["file"], nepochs=conf["nepochs"],batch_size=conf["batch_size"], seed=seed+1 if seed is not None else None, test_sample="012")

    # set seeds to get reproducible results (only if requested)
    if seed is not None:
        try:
            python_random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)
        except:  # deprecated in newer tf versions
            tf.keras.utils.set_random_seed(seed)

    # make model
    model = simple_model_norm(input_dim=conf["input_dim"])
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=conf["learning_rate"]),loss=sqrtR_loss,metrics=[mean_pred])
    model.summary()

    # make callbacks
    callbacks = []
    callbacks.append(tf.keras.callbacks.EarlyStopping(patience=30, mode="min", restore_best_weights=True)) #, monitor="val_loss"))
    # ModelCheckpoint
    checkpoint_filepath = f'./checkpoints/training_{datetime.datetime.now().strftime("%Y.%m.%d.%H.%M.%S")}/' + "cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_filepath)
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(checkpoint_filepath, monitor="val_loss", mode="min", save_best_only=False, save_weights_only=True,))
    # Terminate on NaN such that it is easier to debug
    callbacks.append(tf.keras.callbacks.TerminateOnNaN())
    callbacks.append(tf.keras.callbacks.LearningRateScheduler(scheduler))

    # replace this calculation later:
    nEvents = 15598728 
    nbatch = int(nEvents/conf["batch_size"])

    # train
    history = model.fit(
        train_data_gen,
        epochs=conf["nepochs"],
        callbacks=callbacks,
        verbose=1,
        validation_data=val_data_gen,
        steps_per_epoch=nbatch,
        validation_steps=nbatch,
    )

    # Get training history
    data = pd.DataFrame(history.history)
    log.debug(data.head())

    # Plot loss vs epochs (if nepochs > 1)
    if conf["nepochs"] > 1:
        try:
            from plotting_functions import plot_loss
        except:
            from ML_Keras.plotting_functions import plot_loss
        data['epoch'] = history.epoch
        plot_loss(history, ops.outDir)

    return data  # will be used in CI tests


def options():
    parser = argparse.ArgumentParser()
    # input files
    parser.add_argument("-c", "--conf", help="Configuration file. If provided, all other settings are overruled.", default=None)
    parser.add_argument("-i", "--inFile", help="Input file.", default=None)
    parser.add_argument("-o", "--outDir", help="Output directory for plots", default="./")
    # train settings
    parser.add_argument("-e", "--nepochs", help="Number of epochs.", default=1, type=int)
    parser.add_argument("-b", "--batch_size", help="Training batch size.", default=2048, type=int)
    parser.add_argument("-lr", "--learning_rate", help="Learning rate", default=1e-3, type=float)
    # model settings
    parser.add_argument("-s", "--seed", help="Seed for TensorFlow and NumPy", default=None, type=int)
    parser.add_argument("-ni", "--input_dim", help="Dimension of inputs per event for the first layer.", default=1, type=int)
    parser.add_argument("-ns", "--num_samples", help="Number of events", default=None, type=int)
    return parser.parse_args()

if __name__ == "__main__":
    main()
