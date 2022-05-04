'''
Authors: Anthony Badea, Jonathan Bossio
Date: Monday April 25, 2022
'''

# python imports
import h5py
import tensorflow as tf
import argparse
import json
import datetime
import os
import sys
import logging
import pandas as pd
import random as python_random

# custom imports
try:
    from make_model import make_model
    from get_data import get_data
except:
    from ML_Keras.make_model import make_model
    from ML_Keras.get_data import get_data

# Tensorflow GPU settings
physical_devices = tf.config.list_physical_devices('GPU') 
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def main(config = None):

    # logger
    logging.basicConfig(level = 'INFO', format = '%(levelname)s: %(message)s')
    log = logging.getLogger()

    # user options
    if config is not None:
        conf = config
    else:
        ops = options()
        if ops.conf:
            with open(ops.conf) as f:
                log.info(f"opening {ops.conf}")
                conf = json.load(f)
        else:
            conf = {
                "file": ops.inFile,
                "nepochs": ops.nepochs,
                "train_batch_size": ops.train_batch_size,
                "val_batch_size": ops.val_batch_size,
                "validation_steps" : ops.validation_steps,
                "learning_rate" : ops.learning_rate,
                "input_dim" : ops.input_dim,
                "ndense" : ops.ndense,
                "nnode_per_dense" : ops.nnode_per_dense,
                "seed" : ops.seed,
                "num_samples" : ops.num_samples
            }
            with open('conf.json', 'w') as fp:
                json.dump(conf, fp)

    # protection
    if not conf["file"]:
        log.error('ERROR: No input file was provided, exiting')
        sys.exit(1)

    # training configuration
    if "num_samples" not in conf or conf["num_samples"] is None:  # use size of input sample
        with h5py.File(conf["file"]) as hf:
            conf["num_samples"] = hf["nQuarkJets"]['values'].shape[0]
    conf["train_steps_per_epoch"] = conf["num_samples"] // conf["train_batch_size"]
    log.info("Training configuration: \n" + json.dumps(conf, indent=4, sort_keys=True))

    # data set generators
    seed = None
    if "seed" in conf and conf["seed"] is not None:
        seed = conf["seed"]
    train_data_gen = get_data(conf["file"], conf["nepochs"], conf["train_batch_size"], seed)
    # sample validation data from the same probability density function (but generated val data is statistically independent w.r.t training data)
    val_data_gen = get_data(conf["file"], conf["nepochs"], conf["val_batch_size"], seed+1 if seed is not None else None)

    # set seeds to get reproducible results (only if requested)
    if seed is not None:
        tf.keras.utils.set_random_seed(seed)
        python_random.seed(seed)

    # make model
    model = make_model(input_dim=conf["input_dim"], ndense=conf["ndense"], nnode_per_dense=conf["nnode_per_dense"], learning_rate=conf["learning_rate"])
    model.summary()

    # make callbacks
    callbacks = []
    # EarlyStopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        patience=10, mode="min", restore_best_weights=True, monitor="val_loss"
    )
    callbacks.append(early_stopping)
    # ModelCheckpoint
    checkpoint_filepath = f'./checkpoints/training_{datetime.datetime.now().strftime("%Y.%m.%d.%H.%M.%S")}/' + "cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_filepath)
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_filepath, monitor="val_loss", mode="min", save_best_only=True, save_weights_only=True,
    )
    callbacks.append(model_checkpoint)
    # Terminate on NaN such that it is easier to debug
    callbacks.append(tf.keras.callbacks.TerminateOnNaN())

    # train
    history = model.fit(
        train_data_gen,
        steps_per_epoch=conf["train_steps_per_epoch"],
        epochs=conf["nepochs"],
        callbacks=callbacks,
        verbose=1,
        validation_data=val_data_gen,
        validation_steps=conf["validation_steps"]
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
        plot_loss(history)

    return data  # will be used in CI tests


def options():
    parser = argparse.ArgumentParser()
    # input files
    parser.add_argument("-c",  "--conf", help="Configuration file. If provided, all other settings are overruled.", default=None)
    parser.add_argument("-i",  "--inFile", help="Input file.", default=None)
    # train settings
    parser.add_argument("-ne", "--nepochs", help="Number of epochs.", default=1, type=int)
    parser.add_argument("-tb", "--train_batch_size", help="Training batch size.", default=2048, type=int)
    parser.add_argument("-vb", "--val_batch_size", help="Validation batch size.", default=256, type=int)
    parser.add_argument("-vs", "--validation_steps", help="Number of validation steps.", default=1, type=int)
    parser.add_argument("-lr", "--learning_rate", help="Learning rate", default=1e-3, type=float)
    # model settings
    parser.add_argument("-ni", "--input_dim", help="Dimension of inputs per event for the first layer.", default=1, type=int)
    parser.add_argument("-nl", "--ndense", help="Number of dense layers.", default=1, type=int)
    parser.add_argument("-nd", "--nnode_per_dense", help="Number of nodes per dense layer.", default=30, type=int)
    parser.add_argument("-s", "--seed", help="Seed for TensorFlow and NumPy", default=None, type=int)
    parser.add_argument("-ns", "--num_samples", help="Number of events", default=None, type=int)
    return parser.parse_args()

if __name__ == "__main__":
    main()
