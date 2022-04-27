'''
Author: Anthony Badea
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

# custom imports
from make_model import make_model
from get_data import get_data

# Tensorflow GPU settings
physical_devices = tf.config.list_physical_devices('GPU') 
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def main():

    # logger
    logging.basicConfig(level = 'INFO', format = '%(levelname)s: %(message)s')
    log = logging.getLogger()

    # user options
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
            "tf_seed" : ops.tf_seed
        }
        with open('conf.json', 'w') as fp:
            json.dump(conf, fp)

    # protection
    if not conf["file"]:
        log.error('ERROR: No input file was provided, exiting')
        sys.exit(1)

    # training configuration
    with h5py.File(conf["file"]) as f:
        conf["num_samples"] = f["data"]["ZeroQuarkJetsFlag"].shape[0]
    conf["train_steps_per_epoch"] = conf["num_samples"] // conf["train_batch_size"]
    log.info("Training configuration: \n" + json.dumps(conf, indent=4, sort_keys=True))

    # data set generators
    train_data_gen = get_data(conf["file"], conf["nepochs"], conf["train_batch_size"])
    val_data_gen = get_data(conf["file"], conf["nepochs"], conf["val_batch_size"]) # NOTE: for now we will sample the validation data from the same probability density function

    # set seeds to get reproducible results (only if requested)
    if "tf_seed" in conf and conf["tf_seed"] is not None:
      tf.keras.utils.set_random_seed(conf["tf_seed"])

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
    model.fit(
        train_data_gen,
        steps_per_epoch=conf["train_steps_per_epoch"],
        epochs=conf["nepochs"],
        callbacks=callbacks,
        verbose=1,
        validation_data=val_data_gen,
        validation_steps=conf["validation_steps"]
    )


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
    parser.add_argument("-tfs", "--tf_seed", help="Tensorflow seed", default=None, type=int)
    return parser.parse_args()

if __name__ == "__main__":
    main()
