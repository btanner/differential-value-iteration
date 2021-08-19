import matplotlib as mpl
import numpy as np

mpl.use('Agg')
import matplotlib.pyplot as plt


def run_alg(alg, update_rule, max_iters=50000, epsilon=0.0001):
  if hasattr(alg, update_rule):
    update = getattr(alg, update_rule)
  else:
    print('%s is not implemented', update_rule)
    raise NotImplementedError
  convergence = False
  for i in range(max_iters):
    old_v = alg.v.copy()
    if hasattr(alg, 'r_bar'):
      if type(alg.r_bar) is int:
        old_r_bar = alg.r_bar
      else:
        old_r_bar = alg.r_bar.copy()
    update()
    # print(alg.alpha, alg.beta, alg.v, alg.g)
    if hasattr(alg, 'r_bar'):
      r_bar_error = np.sum(np.abs(old_r_bar - alg.r_bar))
    else:
      r_bar_error = 0
    if np.sum(np.abs(old_v - alg.v)) + r_bar_error < epsilon:
      # print(old_v, alg.v, old_g, alg.g, np.sum(np.abs(old_v - alg.v)), g_error, np.sum(np.abs(old_v - alg.v)) + g_error)
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
  plt.clim(0, 1)
  plt.title(name)
  plt.savefig(name + '.pdf')
  plt.close()
