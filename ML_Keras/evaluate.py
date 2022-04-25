'''
Author: Anthony Badea
Date: Monday April 25, 2022
'''

# python imports
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import tensorflow as tf

# custom code
from get_data import get_full_data
from make_model import make_model

# Tensorflow GPU settings
physical_devices = tf.config.list_physical_devices('GPU') 
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def main():

    # user options
    ops = options()
    if ops.conf:
        with open(ops.conf) as f:
            print(f"opening {ops.conf}")
            conf = json.load(f)
    else:
        conf = {
            "file": ops.inFile,
            "ndense" : ops.ndense,
            "nnode_per_dense" : ops.nnode_per_dense
        }

    # load model
    model = make_model(input_dim=1, ndense=conf["ndense"], nnode_per_dense=conf["nnode_per_dense"], learning_rate=1e-3)
    model.summary()
    model.load_weights(ops.model_weights)
    x, y, normweight = get_full_data(conf["file"])

    # prepare data
    x = x.reshape(x.size, 1)
    y = y.reshape(x.size, 1)
    normweight = normweight.reshape(x.size, 1)
    xa = x[y==0]
    xb = x[y==1]
    bins = np.linspace(0, 8000, 100)
    normweightsa = normweight[y==0]
    normweightsb = normweight[y==1]

    # make model prediction
    p = model.predict(x)

    # plot
    xa = np.multiply(xa, 1000)
    xb = np.multiply(xb, 1000)
    c0, _, _ = plt.hist(xa, bins = bins, alpha = 0.5, weights = normweightsa, label = '#QuarkJets > 0', color = 'red', density=True)
    c1, _, _ = plt.hist(xb, bins = bins, alpha = 0.5, weights = normweightsb, label = '#QuarkJets = 0', color = 'blue', density=True)
    p = np.array(p).reshape(x.size, 1)
    _pp = p[y==0]
    final_weights = (1-_pp)/_pp
    final_weights *= normweightsa
    c2, _, _ = plt.hist(xa, bins = bins, alpha = 0.5, weights = final_weights, label = '#Quarks > 0 reweighted to #QuarksJets = 0', color = 'yellow', density=True) 
    plt.legend()
    plt.show()

def options():
    parser = argparse.ArgumentParser()
    # input files d
    parser.add_argument("-c",  "--conf", help="Configuration file. If provided, all other settings are overruled.", default=None)
    parser.add_argument("-i",  "--inFile", help="Input file.", default=None)
    parser.add_argument("-m",  "--model_weights", help="Model weights.", default=None)
    # model settings
    parser.add_argument("-nl", "--ndense", help="Number of dense layers.", default=1, type=int)
    parser.add_argument("-nd", "--nnode_per_dense", help="Number of nodes per dense layer.", default=30, type=int)
    return parser.parse_args()

if __name__ == "__main__":
    main()