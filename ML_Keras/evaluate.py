# python imports
import numpy as np
import matplotlib.pyplot as plt

# custom code
from get_data import get_full_data
from make_model import make_model

def evaluate(model, x, y, normweight):

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

if __name__ == "__main__":
	model, callbacks = make_model(input_dim=1, nodes=30, learning_rate=1e-3)
	model.load_weights("best_model.h5")
	Xs, ys, wgts = get_full_data('mc16a_dijets_JZAll_for_reweighting.h5')
	evaluate(model, Xs, ys, wgts)