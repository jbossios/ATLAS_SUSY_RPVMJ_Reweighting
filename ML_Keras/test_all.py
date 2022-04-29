def test_train(update_ref = False):

    import os
    import sys
    import pandas as pd
    
    # Import train and evaluate modules
    from train import main as train
    from train import options

    ops = options()
    conf = {
        "file": "/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/reweighting/Jona/H5_files/v3/mc16a_dijets_JZAll_for_reweighting.h5",
        "nepochs": 1,
	"num_samples": ops.train_batch_size * 2,
        "train_batch_size": ops.train_batch_size,
        "val_batch_size": ops.val_batch_size,
        "validation_steps" : ops.validation_steps,
        "learning_rate" : ops.learning_rate,
        "input_dim" : ops.input_dim,
        "ndense" : ops.ndense,
        "nnode_per_dense" : ops.nnode_per_dense,
        "seed" : 1
    }
    data = train(conf)[['loss', 'accuracy']]  # Temporary until I get reproducible val results
    if update_ref:  # save dataframe
        data.to_csv('train.ref', index=False)
    # Load reference and check if result agrees with it
    ref = pd.read_csv('train.ref')
    diff = data.compare(ref)
    if diff.size != 0:
      print("ERROR: test result doesn't match the reference (train.ref), if differences are expected/understood, update the reference")
      print("Reference dataframe:")
      print(ref.head())
      print("New dataframe:")
      print(data.head())
      sys.exit(1)

if __name__ == '__main__':
    test_train(False)
