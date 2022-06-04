'''
Authors: Jonathan Bossio, Anthony Badea
Date: Monday April 25, 2022
'''

import tensorflow as tf
import numpy as np

def make_model(**kargs):
    
    # Create model
    model = tf.keras.Sequential()
    
    nodes = [kargs["nnode_per_dense"] for i in range(kargs["ndense"]-1)] + [1]
    input_dims = [kargs["input_dim"]] + nodes
    activations = ["relu" for i in range(kargs["ndense"]-1)] + ["sigmoid"]
    for node, input_dim, activation in zip(nodes, input_dims, activations):
        model.add(tf.keras.layers.Dense(node, input_dim=input_dim, activation=activation))
    
    # Compile model
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=kargs["learning_rate"]),
        loss=kargs["loss"], #tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    model, _ = make_model(input_dim=10, nodes=100, learning_rate=1e4)
    model.summary()
