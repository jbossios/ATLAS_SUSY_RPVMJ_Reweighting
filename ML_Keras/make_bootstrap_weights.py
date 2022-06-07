
import numpy as np
import scipy.stats
import tensorflow as tf
import argparse
import os

from make_model import simple_model

def main():

	# user options
	ops = options()

	# get list of checkpoint directories
	dir_list = ['./bootstrap_2022.06.07.18.13.47/training_0/'] #handleInput()
	if not dir_list:
		print("No entries in the dir_list")
		exit()

	# use the first checkpoint to infer the weights
	reader = tf.train.load_checkpoint(dir_list[0])
	shape_from_key = reader.get_variable_to_shape_map()
	kernels = [i for i in sorted(shape_from_key.keys()) if "kernel" in i and "optimizer" not in i]
	biases = [i for i in sorted(shape_from_key.keys()) if "bias" in i and "optimizer" not in i]
	keys = kernels + biases
	print(keys)

	# loop over dir and append to list
	weights = {key:[] for key in keys}
	for check in dir_list:
		reader = tf.train.load_checkpoint(check)
		for key in keys:
			weights[key].append(reader.get_tensor(key))

	# concatenate and take median +- 1/2 * IQR
	bootstrap_weights = {
		'w_nom' : {},
		'w_up' : {},
		'w_down' : {}
	}
	for key, val in weights.items():
		x = np.stack(val,0)
		iqr = scipy.stats.iqr(x,0)
		median = np.median(x,0)
		bootstrap_weights['w_nom'][key] = median
		bootstrap_weights['w_up'][key] = median + 0.5 * iqr
		bootstrap_weights['w_down'][key] = median - 0.5 * iqr

	# save models
	model = simple_model(input_dim=5)
	for bootstrap_key, bootstrap_weight in bootstrap_weights.items():
		for iL, (kernel, bias) in enumerate(zip(kernels, biases)):
			model.layers[iL+1].set_weights([bootstrap_weight[kernel],bootstrap_weight[bias]])
		model.save_weights(f'./test/{bootstrap_key}/{bootstrap_key}.ckpt')




def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",  "--inFile", help="Input file.", default=None)
    parser.add_argument("-o",  "--outDir", help="Output directory", default="./")
    return parser.parse_args()

if __name__ == "__main__":
	main()





