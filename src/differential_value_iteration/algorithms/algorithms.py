import numpy as np


def check_mrp(mrp):
  if mrp.transitions.ndim != 2:
    raise ValueError(
        f'mrp transitions should be 2 dimensional, not: {mrp.transitions.shape}')
  if mrp.rewards.ndim != 1:
    raise ValueError(
        f'mrp rewards should be 1 dimensional, not: {mrp.rewards.shape}')


def check_mdp(mdp):
  if mdp.transitions.ndim != 3:
    raise ValueError(
        f'mrd transitions should be 3 dimensional, not: {mdp.transitions.shape}')
  if mdp.rewards.ndim != 2:
    raise ValueError(
        f'mrd rewards should be 2 dimensional, not: {mdp.rewards.shape}')


class RVI_Evaluation:
  def __init__(self, mrp, v, alpha: float = 1., ref_idx: int = 0):
    check_mrp(mrp)
    self.p = mrp.transitions
    self.r = mrp.rewards
    self.init_v = v.copy()
    self.alpha = alpha
    self.idx = 0
    self.ref_idx = ref_idx
    self.num_states = mrp.num_states
    self.reset()

  def exec_sync(self):
    self.v += self.alpha * (
        self.r + np.dot(self.p,
                        self.v - self.v[self.ref_idx]) - self.v)
    return

  def exec_async(self):
    idx = self.idx
    self.v[idx] += self.alpha * (
        self.r[idx] + np.dot(self.p[idx], self.v - self.v[self.ref_idx]) -
        self.v[idx])
    self.idx = (self.idx + 1) % self.num_states
    return

  def reset(self):
    self.v = self.init_v.copy()


class RVI_Control(object):
  def __init__(self, mdp, v, alpha=1.0, ref_idx=0):
    check_mdp(mdp)
    self.p = mdp.transitions
    self.r = mdp.rewards
    self.init_v = v.copy()
    self.alpha = alpha
    self.idx = 0
    self.ref_idx = ref_idx
    self.num_states = mdp.num_states
    self.num_actions = mdp.num_actions
    self.reset()

  def exec_sync(self):
    temp_s_by_a = np.zeros((self.num_states, self.num_actions))
    for a in range(self.num_actions):
      temp_s_by_a[:, a] = self.r[a] + np.dot(self.p[a], self.v - self.v[
        self.ref_idx]) - self.v
    delta = np.max(temp_s_by_a, axis=1)
    self.v += self.alpha * delta
    return

  def exec_async(self):
    idx = self.idx
    temp_a = np.zeros(self.num_actions)
    for a in range(self.num_actions):
      temp_a[a] = self.r[a][idx] + np.dot(self.p[a][idx],
                                          self.v - self.v[self.ref_idx]) - \
                  self.v[idx]
    delta = np.max(temp_a)
    self.v[idx] += self.alpha * delta
    self.idx = (self.idx + 1) % self.num_states
    return

  def reset(self):
    self.v = self.init_v.copy()


class DVI_Evaluation(object):
  def __init__(self, mrp, init_v, init_r_bar, alpha=1.0, beta=1.0):
    check_mrp(mrp)
    self.p = mrp.transitions
    self.r = mrp.rewards
    self.init_v = init_v.copy()
    self.init_r_bar = init_r_bar
    self.alpha = alpha
    self.beta = beta
    self.idx = 0
    self.num_states = mrp.num_states
    self.reset()

  def exec_sync(self):
    delta = self.r - self.r_bar + np.dot(self.p, self.v) - self.v
    self.v += self.alpha * delta
    self.r_bar += self.beta * np.sum(delta)
    return

  def exec_async(self):
    idx = self.idx
    delta = self.r[idx] - self.r_bar + np.dot(self.p[idx], self.v) - self.v[idx]
    self.v[idx] += self.alpha * delta
    self.r_bar += self.beta * delta
    self.idx = (self.idx + 1) % self.num_states
    return

  def reset(self):
    self.v = self.init_v.copy()
    self.r_bar = self.init_r_bar


class DVI_Control(object):
  def __init__(self, mdp, init_v, init_r_bar, alpha=1.0, beta=1.0):
    check_mdp(mdp)
    self.p = mdp.transitions
    self.r = mdp.rewards
    self.init_v = init_v.copy()
    self.init_r_bar = init_r_bar
    self.alpha = alpha
    self.beta = beta
    self.idx = 0
    self.num_actions = mdp.num_actions
    self.num_states = mdp.num_states
    self.reset()

  def exec_sync(self):
    temp_s_by_a = np.zeros((self.num_states, self.num_actions))
    for a in range(self.num_actions):
      temp_s_by_a[:, a] = self.r[a] - self.r_bar + np.dot(self.p[a],
                                                          self.v) - self.v
    delta = np.max(temp_s_by_a, axis=1)
    self.v += self.alpha * delta
    self.r_bar += self.beta * np.sum(delta)
    return

  def exec_async(self):
    idx = self.idx
    temp_a = np.zeros(self.num_actions)
    for a in range(self.num_actions):
      temp_a[a] = self.r[a][idx] - self.r_bar + np.dot(self.p[a][idx], self.v) - \
                  self.v[idx]
    delta = np.max(temp_a)
    self.v[idx] += self.alpha * delta
    self.r_bar += self.beta * delta
    self.idx = (self.idx + 1) % self.num_states
    return

  def reset(self):
    self.v = self.init_v.copy()
    self.r_bar = self.init_r_bar


class MDVI_Evaluation(object):
  def __init__(self, mrp, init_v, init_r_bar, alpha=1.0, beta=1.0):
    check_mrp(mrp)
    self.p = mrp.transitions
    self.r = mrp.rewards
    self.init_v = init_v.copy()
    self.init_r_bar = init_r_bar.copy()
    self.alpha = alpha
    self.beta = beta
    self.num_states = mrp.num_states
    self.idx = 0
    self.reset()

  def exec_sync(self):
    self.r_bar = np.dot(self.p, self.r_bar)
    delta = self.r - self.r_bar + np.dot(self.p, self.v) - self.v
    self.v += self.alpha * delta
    self.r_bar += self.beta * delta
    return

  def exec_async(self):
    idx = self.idx
    self.r_bar[idx] = np.dot(self.p[idx], self.r_bar)
    delta = self.r[idx] - self.r_bar[idx] + np.dot(self.p[idx], self.v) - \
            self.v[idx]
    self.v[idx] += self.alpha * delta
    self.r_bar[idx] += self.beta * delta
    self.idx = (self.idx + 1) % self.num_states
    return

  def reset(self):
    self.v = self.init_v.copy()
    self.r_bar = self.init_r_bar.copy()


class MDVI_Control1(object):
  def __init__(self, mdp, init_v, init_r_bar, alpha=1.0, beta=1.0,
      threshold=0.01):
    check_mdp(mdp)
    self.p = mdp.transitions
    self.r = mdp.rewards
    self.init_v = init_v.copy()
    self.init_r_bar = init_r_bar.copy()
    self.alpha = alpha
    self.beta = beta
    self.num_states = mdp.num_states
    self.num_actions = mdp.num_actions
    self.idx = 0
    self.threshold = threshold
    self.reset()

  def exec_sync(self):
    num_states = self.num_states
    num_actions = self.num_actions
    temp_s_by_a = np.zeros((num_states, num_actions))
    for a in range(num_actions):
      temp_s_by_a[:, a] = np.dot(self.p[a], self.r_bar)
    self.r_bar = np.max(temp_s_by_a, axis=1)
    delta = np.zeros(num_states)
    for s in range(num_states):
      max_actions = np.where(temp_s_by_a[s] > self.r_bar[s] - self.threshold)[0]
      temp_a = np.zeros(len(max_actions))
      for i in range(len(max_actions)):
        temp_a[i] = self.r[max_actions[i]][s] - self.r_bar[s] + np.dot(
            self.p[max_actions[i]][s], self.v) - self.v[s]
      delta[s] = np.max(temp_a)
    self.v += self.alpha * delta
    self.r_bar += self.beta * delta
    return

  def exec_async(self):
    idx = self.idx
    num_actions = self.num_actions
    temp_a = np.zeros(num_actions)
    for a in range(num_actions):
      temp_a[a] = np.dot(self.p[a][idx], self.r_bar)
    self.r_bar[idx] = np.max(temp_a)
    max_actions = np.where(temp_a > self.r_bar[idx] - self.threshold)[0]
    temp_a = np.zeros(len(max_actions))
    for i in range(len(max_actions)):
      temp_a[i] = self.r[max_actions[i]][idx] - self.r_bar[idx] + np.dot(
          self.p[max_actions[i]][idx], self.v) - self.v[idx]
    delta = np.max(temp_a)
    self.v[idx] += self.alpha * delta
    self.r_bar[idx] += self.beta * delta
    self.idx = (self.idx + 1) % self.num_states
    return

  def reset(self):
    self.v = self.init_v.copy()
    self.r_bar = self.init_r_bar.copy()


class MDVI_Control2(object):
  def __init__(self, mdp, init_v, init_r_bar, alpha=1.0, beta=1.0):
    check_mdp(mdp)
    self.p = mdp.transitions
    self.r = mdp.rewards
    self.init_v = init_v.copy()
    self.init_r_bar = init_r_bar.copy()
    self.alpha = alpha
    self.beta = beta
    self.num_states = mdp.num_states
    self.num_actions = mdp.num_actions
    self.idx = 0
    self.reset()

  def exec_sync(self):
    num_states = self.num_states
    num_actions = self.num_actions
    temp_s_by_a = np.zeros((num_states, num_actions))
    for a in range(num_actions):
      temp_s_by_a[:, a] = np.dot(self.p[a], self.r_bar)
    self.r_bar = np.max(temp_s_by_a, axis=1)
    temp_s_by_a = np.zeros((num_states, num_actions))
    for a in range(num_actions):
      temp_s_by_a[:, a] = self.r[a] - self.r_bar + np.dot(self.p[a],
                                                          self.v) - self.v
    delta = np.max(temp_s_by_a, axis=1)
    self.v += self.alpha * delta
    self.r_bar += self.beta * delta
    return

  def exec_async(self):
    idx = self.idx
    num_actions = self.num_actions
    temp_a = np.zeros(num_actions)
    for a in range(num_actions):
      temp_a[a] = np.dot(self.p[a][idx], self.r_bar)
    self.r_bar[idx] = np.max(temp_a)
    temp_a = np.zeros(num_actions)
    for a in range(num_actions):
      temp_a[a] = self.r[a][idx] - self.r_bar[idx] + np.dot(self.p[a][idx],
                                                            self.v) - self.v[
                    idx]
    delta = np.max(temp_a)
    self.v[idx] += self.alpha * delta
    self.r_bar[idx] += self.beta * delta
    self.idx = (self.idx + 1) % self.num_states
    return

  def reset(self):
    self.v = self.init_v.copy()
    self.r_bar = self.init_r_bar.copy()
