

# python imports
import h5py

# custom imports
from make_model import make_model
from get_data import get_data

def main():

	# load model
	model, callbacks = make_model(input_dim=1, nodes=100, learning_rate=1e-3)
	model.summary()

	# load data
	file = 'mc16a_dijets_JZAll_for_reweighting.h5'
	with h5py.File(file) as f:
		num_samples = f['data']['ZeroQuarkJetsFlag'].shape[0]
	batch_size = 2048
	nepochs = 3
	datagen = get_data(file, nepochs, batch_size)
	print(f"Num samples {num_samples}, Epochs {nepochs}, Batch size {batch_size}, Steps per epoch {num_samples // batch_size}")
	# train
	model.fit(datagen, 
			steps_per_epoch= num_samples // batch_size,
			epochs = nepochs,
			callbacks = callbacks,
			verbose=1) #,
			# validation_data = datagen)

if __name__ == "__main__":
	main()