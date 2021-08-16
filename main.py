"""Sample program that runs a dvi algorithm on a sample MDP."""
import numpy as np

from differential_value_iteration.algorithms import algorithms
from differential_value_iteration.environments import environments
 
def run():
  env = environments.ThreeLoopMRP()
  alphas = np.arange(0.01, 1.01, 0.01)
  convergence_flags = np.zeros(alphas.shape)
  for _, alpha in enumerate(alphas):
    print(f'Starting alpha:{alpha}', end=' ')
    _, convergence,_ = algorithms.rvi_sync(3, env.P, env.R, max_iters=1000, ref_state=1, alpha=alpha) 
    print(f'Converged? {convergence}')

if __name__ == '__main__':
  run()
