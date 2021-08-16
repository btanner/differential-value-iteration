import numpy as np

from src.differential_value_iteration.environments import MRP, MDP


class RVI_Evaluation(object):
    def __init__(self, mrp, v, alpha=1.0, ref_idx=0):
        assert type(mrp) is MRP
        self.P = mrp.P.copy()
        self.r = mrp.r.copy()
        self.init_v = v.copy()
        self.alpha = alpha
        self.idx = 0
        self.ref_idx = ref_idx
        self.num_states = mrp.num_states()
        self.reset()
    
    def exec_sync(self):
        self.v += self.alpha * (self.r + np.dot(self.P, self.v - self.v[self.ref_idx]) - self.v)
        return
    
    def exec_async(self):
        idx = self.idx
        self.v[idx] += self.alpha * (self.r[idx] + np.dot(self.P[idx], self.v - self.v[self.ref_idx]) - self.v[idx])
        self.idx = (self.idx + 1) % self.num_states
        return
    
    def reset(self):
        self.v = self.init_v.copy()


class RVI_Control(object):
    def __init__(self, mdp, v, alpha=1.0, ref_idx=0):
        assert type(mdp) is MDP
        self.P = mdp.P.copy()
        self.r = mdp.r.copy()
        self.init_v = v.copy()
        self.alpha = alpha
        self.idx = 0
        self.ref_idx = ref_idx
        self.num_states = mdp.num_states()
        self.num_actions = mdp.num_actions()
        self.reset()
    
    def exec_sync(self):
        temp_s_by_a = np.zeros((self.num_states, self.num_actions))
        for a in range(self.num_actions):
            temp_s_by_a[:, a] = self.r[a] + np.dot(self.P[a], self.v - self.v[self.ref_idx]) - self.v
        delta = np.max(temp_s_by_a, axis=1)
        self.v += self.alpha * delta
        return
    
    def exec_async(self):
        idx = self.idx
        temp_a = np.zeros(self.num_actions)
        for a in range(self.num_actions):
            temp_a[a] = self.r[a][idx] + np.dot(self.P[a][idx], self.v - self.v[self.ref_idx]) - self.v[idx]
        delta = np.max(temp_a)
        self.v[idx] += self.alpha * delta
        self.idx = (self.idx + 1) % self.num_states
        return
    
    def reset(self):
        self.v = self.init_v.copy()


class DVI_Evaluation(object):
    def __init__(self, mrp, v, g, alpha=1.0, beta=1.0):
        assert type(mrp) is MRP
        self.P = mrp.P.copy()
        self.r = mrp.r.copy()
        self.init_v = v.copy()
        self.init_g = g
        self.alpha = alpha
        self.beta = beta
        self.idx = 0
        self.num_states = mrp.num_states()
        self.reset()
    
    def exec_sync(self):
        delta = self.r - self.g + np.dot(self.P, self.v) - self.v
        self.v += self.alpha * delta
        self.g += self.beta * np.sum(delta)
        return
    
    def exec_async(self):
        idx = self.idx
        delta = self.r[idx] - self.g + np.dot(self.P[idx], self.v) - self.v[idx]
        self.v[idx] += self.alpha * delta
        self.g += self.beta * delta
        self.idx = (self.idx + 1) % self.num_states
        return
    
    def reset(self):
        self.v = self.init_v.copy()
        self.g = self.init_g


class DVI_Control(object):
    def __init__(self, mdp, v, g, alpha=1.0, beta=1.0):
        assert type(mdp) is MDP
        self.P = mdp.P.copy()
        self.r = mdp.r.copy()
        self.init_v = v.copy()
        self.init_g = g
        self.alpha = alpha
        self.beta = beta
        self.idx = 0
        self.num_actions = mdp.num_actions()
        self.num_states = mdp.num_states()
        self.reset()
    
    def exec_sync(self):
        temp_s_by_a = np.zeros((self.num_states, self.num_actions))
        for a in range(self.num_actions):
            temp_s_by_a[:, a] = self.r[a] - self.g + np.dot(self.P[a], self.v) - self.v
        delta = np.max(temp_s_by_a, axis=1)
        self.v += self.alpha * delta
        self.g += self.beta * np.sum(delta)
        return
    
    def exec_async(self):
        idx = self.idx
        temp_a = np.zeros(self.num_actions)
        for a in range(self.num_actions):
            temp_a[a] = self.r[a][idx] - self.g + np.dot(self.P[a][idx], self.v) - self.v[idx]
        delta = np.max(temp_a)
        self.v[idx] += self.alpha * delta
        self.g += self.beta * delta
        self.idx = (self.idx + 1) % self.num_states
        return
    
    def reset(self):
        self.v = self.init_v.copy()
        self.g = self.init_g


class MDVI_Evaluation(object):
    def __init__(self, mrp, v, g, alpha=1.0, beta=1.0):
        assert type(mrp) is MRP
        self.P = mrp.P.copy()
        self.r = mrp.r.copy()
        self.init_v = v.copy()
        self.init_g = g.copy()
        self.alpha = alpha
        self.beta = beta
        self.num_states = mrp.num_states()
        self.idx = 0
        self.reset()
    
    def exec_sync(self):
        self.g = np.dot(self.P, self.g)
        delta = self.r - self.g + np.dot(self.P, self.v) - self.v
        self.v += self.alpha * delta
        self.g += self.beta * delta
        return
    
    def exec_async(self):
        idx = self.idx
        self.g[idx] = np.dot(self.P[idx], self.g)
        delta = self.r[idx] - self.g[idx] + np.dot(self.P[idx], self.v) - self.v[idx]
        self.v[idx] += self.alpha * delta
        self.g[idx] += self.beta * delta
        self.idx = (self.idx + 1) % self.num_states
        return
    
    def reset(self):
        self.v = self.init_v.copy()
        self.g = self.init_g.copy()


class MDVI_Control2(object):
    def __init__(self, mdp, v, g, alpha=1.0, beta=1.0):
        assert type(mdp) is MDP
        self.P = mdp.P.copy()
        self.r = mdp.r.copy()
        self.init_v = v.copy()
        self.init_g = g.copy()
        self.alpha = alpha
        self.beta = beta
        self.num_states = mdp.num_states()
        self.num_actions = mdp.num_actions()
        self.idx = 0
        self.reset()
    
    def exec_sync(self):
        num_states = self.num_states
        num_actions = self.num_actions
        temp_s_by_a = np.zeros((num_states, num_actions))
        for a in range(num_actions):
            temp_s_by_a[:, a] = np.dot(self.P[a], self.g)
        self.g = np.max(temp_s_by_a, axis=1)
        temp_s_by_a = np.zeros((num_states, num_actions))
        for a in range(num_actions):
            temp_s_by_a[:, a] = self.r[a] - self.g + np.dot(self.P[a], self.v) - self.v
        delta = np.max(temp_s_by_a, axis=1)
        self.v += self.alpha * delta
        self.g += self.beta * delta
        return
    
    def exec_async(self):
        idx = self.idx
        num_actions = self.num_actions
        temp_a = np.zeros(num_actions)
        for a in range(num_actions):
            temp_a[a] = np.dot(self.P[a][idx], self.g)
        self.g[idx] = np.max(temp_a)
        temp_a = np.zeros(num_actions)
        for a in range(num_actions):
            temp_a[a] = self.r[a][idx] - self.g[idx] + np.dot(self.P[a][idx], self.v) - self.v[idx]
        delta = np.max(temp_a)
        self.v[idx] += self.alpha * delta
        self.g[idx] += self.beta * delta
        self.idx = (self.idx + 1) % self.num_states
        return
    
    def reset(self):
        self.v = self.init_v.copy()
        self.g = self.init_g.copy()