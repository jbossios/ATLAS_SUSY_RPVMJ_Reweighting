'''
Author: Anthony Badea
Date: Monday April 25, 2022
'''

import tensorflow as tf
import numpy as np

def make_model(**kargs):
    
    # Create model
    model = tf.keras.Sequential()
    for i in range(kargs["ndense"]):
        model.add(tf.keras.layers.Dense(kargs["nnode_per_dense"], input_dim=kargs["input_dim"], activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    
    # Compile model
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=kargs["learning_rate"]),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    model, _ = make_model(input_dim=10, nodes=100, learning_rate=1e4)
    model.summary()
