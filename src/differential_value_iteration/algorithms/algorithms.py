import numpy as np

# helper functions

def span(x):
    "computes and returns the span of a given vector x: sp(x) = max(x) - min(x)"
    return abs(np.max(x) - np.min(x))

# algorithms

def rvi_sync(num_states, P, R, max_iters=1000, ref_state=0, alpha=1.0, epsilon=0.01,
             v_init=0):
    """
    Runs the synchronous Relative Value Iteration algorithm (Puterman 1994, Section 8.5.5) for given Markov chain dyanmics

        Parameters:
            num_states (int) : number of states in the underlying MDP who's values to estimate
            P (ndarray) : the |S| x |S| transition matrix
            R (ndarray) : the |S| x 1 one-step expected reward vector
            max_iters (int) : the maximum number of iterations to run the algorithm
            ref_state (int) : RVI requires the specification of a reference state
            alpha (float) : the step size for the value estimates
            epsilon (float) : the parameter to check for convergence
            v_init (float or ndarray) : initial value estimates, if other than all zeros

        Returns:
                v (ndarray) : value estimates
                convergence (bool) : if the algorithm converged before max_iters
                t+1 (int) : min(max_iters, number of iterations the algorithm ran before convergence)
    """
    v = np.zeros((max_iters, num_states)) + v_init
    t = 0
    convergence = False
    while t < max_iters-1:
        delta = R - (np.ones(num_states) * v[t][ref_state]) + np.matmul(P,v[t]) - v[t]
        v[t+1] = v[t] + alpha * delta
        if span(v[t+1] - v[t]) < epsilon:
            convergence = True
            break
        t += 1
    return v, convergence, t+1

# Differential Value Iteration (new)

def dvi_sync(num_states, P, R, max_iters=1000, alpha=1.0, beta=None, epsilon=0.01,
             v_init=0, r_bar_init=0):
    """
    Runs the synchronous Differential Value Iteration algorithm (new) for given Markov chain dyanmics

        Parameters:
            num_states (int) : number of states in the underlying MDP who's values to estimate
            P (ndarray) : the |S| x |S| transition matrix
            R (ndarray) : the |S| x 1 one-step expected reward vector
            max_iters (int) : the maximum number of iterations to run the algorithm
            ref_state (int) : RVI requires the specification of a reference state
            alpha (float) : the step size for the value estimates
            beta (float) : the step size for the reward-rate estimate
            epsilon (float) : the parameter to check for convergence
            v_init (float or ndarray) : initial value estimates, if other than all zeros
            r_bar_init (float) : initial reward-rate estimate, if other than zero

        Returns:
                v (ndarray) : value estimates
                convergence (bool) : if the algorithm converged before max_iters
                t+1 (int) : min(max_iters, number of iterations the algorithm ran before convergence)
    """
    if beta is None:    # in case beta is unspecified
        beta = 1 - alpha
    v = np.zeros((max_iters, num_states)) + v_init
    r_bar = np.ones((max_iters, num_states)) * r_bar_init
    t = 0
    convergence = False
    while t < max_iters-1:
        delta = R - r_bar[t] + np.matmul(P,v[t]) - v[t]
        v[t+1] = v[t] + alpha * delta
        r_bar[t+1] = r_bar[t] + beta * np.mean(delta)
        if span(v[t+1] - v[t]) < epsilon:   # convergence criteria should take r_bar into account as well
            convergence = True
            break
        t += 1
    return v, convergence, t+1
