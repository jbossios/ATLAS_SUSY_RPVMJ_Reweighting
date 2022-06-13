'''
Authors: Anthony Badea, Jonathan Bossio
Date: Monday June 13, 2022
'''

# python imports
import h5py
import numpy as np
import argparse
import os

# custom code
try:
    from make_model import simple_model, sqrtR_loss, mean_pred
except:
    from ML_Keras.make_model import simple_model, sqrtR_loss, mean_pred

# Tensorflow GPU settings
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def main(config=None):

    # user options
    ops = options()

    # load file
    with h5py.File(conf["file"], "r") as f:
        # pick up variables from file
        x = np.stack([
            np.array(f['EventVars']['HT']),
            np.array(f['EventVars']['deta']),
            np.array(f['EventVars']['djmass']),
            np.array(f['EventVars']['minAvgMass']),
            np.array(f['source']['pt'][:, 0])
        ], -1)

    # load model
    model = simple_model(input_dim=RegA_x.shape[1])
    model.compile(loss=sqrtR_loss, metrics=[mean_pred])
    model.summary()

    # if checkpoint directory provided use the latest
    if os.path.isdir(ops.model_weights):
        latest = tf.train.latest_checkpoint(ops.model_weights)
        print(f"Using latest weights from checkpoint directory: {latest}")
        model.load_weights(latest).expect_partial()
    elif ops.model_weights == "1":
        latest = tf.train.latest_checkpoint(glob.glob("checkpoints/*")[-1])
        print(f"Using latest weights from checkpoint directory: {latest}")
        model.load_weights(latest).expect_partial()
    else:
        model.load_weights(ops.model_weights).expect_partial()

    # make prediction
        pred = model.predict(x, batch_size=10000).flatten()

        # save to file
        np.save(os.path.join(ops.outDir, "pred.npy"), pred)


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",  "--inFile", help="Input file.",
                        default=None, required=True)
    parser.add_argument("-o",  "--outDir",
                        help="Output directory", default="./")
    parser.add_argument("-m",  "--model_weights",
                        help="Model weights.", default=None, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
