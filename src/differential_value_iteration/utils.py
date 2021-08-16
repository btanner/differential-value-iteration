import numpy as np
import matplotlib.pyplot as plt


def run_alg(alg, update_rule, max_iters=50000, epsilon=0.001):
	if hasattr(alg, update_rule):
		update = getattr(alg, update_rule)
	else:
		print('%s is not implemented', update_rule)
		raise NotImplementedError
	convergence = False
	for i in range(max_iters):
		old_v = alg.v.copy()
		update()
		if np.sum(np.abs(old_v - alg.v)) < epsilon:   # convergence criteria should take r_bar into account as well
			convergence = True
			break
	return convergence


def draw(results, name, alpha_list, beta_list=None):
	plt.figure(figsize=(15, 15))
	plt.yticks(np.arange(alpha_list.__len__()), alpha_list)
	plt.ylabel(r'$\alpha$', rotation=0, labelpad=20)
	if beta_list is None:
		results = np.array([results]).reshape(-1, 1)
	else:
		plt.xlabel(r'$\beta$')
		plt.xticks(np.arange(beta_list.__len__()), beta_list)
	
	plt.imshow(results, cmap='viridis', interpolation='nearest')
	plt.colorbar()
	plt.title(name)
	plt.savefig(name+'.pdf')