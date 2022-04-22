

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
	batch_size = 256 #2048
	nepochs = 2
	datagen = get_data('mc16a_dijets_JZAll_for_reweighting.h5', nepochs, batch_size)

	# train
	model.fit(datagen, 
			epochs = nepochs,
			callbacks = callbacks,
			verbose=1) #,
			# validation_data = datagen)

if __name__ == "__main__":
	main()