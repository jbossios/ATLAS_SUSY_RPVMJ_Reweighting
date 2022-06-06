'''
Authors: Jonathan Bossio, Anthony Badea
Date: Monday April 25, 2022
'''

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

def myacc(y_true,y_pred):
    y_true = tf.gather(y_true, [0], axis=1) # actual y_true for loss
    return tf.reduce_sum(tf.cast((y_true - y_pred)<=0.1, tf.int32))/tf.size(y_true)

def make_model(**kargs):
    
    # Create model
    # model = tf.keras.Sequential()
    # nodes = [kargs["nnode_per_dense"] for i in range(kargs["ndense"]-1)] + [1]
    # input_dims = [kargs["input_dim"]] + nodes
    # activations = ["relu" for i in range(kargs["ndense"]-1)] + ["sigmoid"]
    # for node, input_dim, activation in zip(nodes, input_dims, activations):
    #     model.add(tf.keras.layers.Dense(node, input_dim=input_dim, activation=activation))
    
    inputs = Input((kargs["input_dim"], ))
    hidden_layer_1 = Dense(50, activation='relu')(inputs)
    hidden_layer_2 = Dense(100, activation='relu')(hidden_layer_1)
    hidden_layer_3 = Dense(50, activation='relu')(hidden_layer_2)
    outputs = Dense(1, activation='sigmoid')(hidden_layer_3)
    model = Model(inputs=inputs, outputs=outputs)

    # Compile model
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=kargs["learning_rate"]),
        loss=kargs["loss"], #tf.keras.losses.BinaryCrossentropy(),
        metrics=[myacc],
    )

    return model


if __name__ == "__main__":
    model, _ = make_model(input_dim=10, nodes=100, learning_rate=1e4)
    model.summary()
