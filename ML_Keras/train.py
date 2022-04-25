# python imports
import h5py
import tensorflow as tf

# custom imports
from make_model import make_model
from get_data import get_data

# Tensorflow GPU settings
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def main():

    # training configuration
    conf = {
        "file": "mc16a_dijets_JZAll_for_reweighting.h5",
        "nepochs": 5,
        "train_batch_size": 2048,
        "val_batch_size": 256,
        "validation_steps" : 10,
        # 
        "learning_rate" : 1e-3,
        "input_dim" : 1,
        "nodes" : 30,
    }
    with h5py.File(conf["file"]) as f:
        conf["num_samples"] = f["data"]["ZeroQuarkJetsFlag"].shape[0]
    conf["train_steps_per_epoch"] = conf["num_samples"] // conf["train_batch_size"]
    print(conf)

    # data set generators
    train_data_gen = get_data(conf["file"], conf["nepochs"], conf["train_batch_size"])
    val_data_gen = get_data(conf["file"], conf["nepochs"], conf["val_batch_size"]) # to-do be more careful about partitioning the train and val data

    # make model
    model, callbacks = make_model(input_dim=1, nodes=30, learning_rate=conf["learning_rate"])
    model.summary()

    # train
    model.fit(
        train_data_gen,
        steps_per_epoch=conf["train_steps_per_epoch"],
        epochs=conf["nepochs"],
        callbacks=callbacks,
        verbose=1,
        validation_data=val_data_gen,
        validation_steps=1
    )


if __name__ == "__main__":
    main()
