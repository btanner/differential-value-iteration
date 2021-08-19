import numpy as np

# from differential_value_iteration.environments import MRP, MDP


class RVI_Evaluation(object):
    def __init__(self, mrp, v, alpha=1.0, ref_idx=0):
        # assert type(mrp) is MRP
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
        # assert type(mdp) is MDP
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
    def __init__(self, mrp, init_v, init_r_bar, alpha=1.0, beta=1.0):
        # assert type(mrp) is MRP
        self.P = mrp.P.copy()
        self.r = mrp.r.copy()
        self.init_v = init_v.copy()
        self.init_r_bar = init_r_bar
        self.alpha = alpha
        self.beta = beta
        self.idx = 0
        self.num_states = mrp.num_states()
        self.reset()
    
    def exec_sync(self):
        delta = self.r - self.r_bar + np.dot(self.P, self.v) - self.v
        self.v += self.alpha * delta
        self.r_bar += self.beta * np.sum(delta)
        return
    
    def exec_async(self):
        idx = self.idx
        delta = self.r[idx] - self.r_bar + np.dot(self.P[idx], self.v) - self.v[idx]
        self.v[idx] += self.alpha * delta
        self.r_bar += self.beta * delta
        self.idx = (self.idx + 1) % self.num_states
        return
    
    def reset(self):
        self.v = self.init_v.copy()
        self.r_bar = self.init_r_bar


class DVI_Control(object):
    def __init__(self, mdp, init_v, init_r_bar, alpha=1.0, beta=1.0):
        # assert type(mdp) is MDP
        self.P = mdp.P.copy()
        self.r = mdp.r.copy()
        self.init_v = init_v.copy()
        self.init_r_bar = init_r_bar
        self.alpha = alpha
        self.beta = beta
        self.idx = 0
        self.num_actions = mdp.num_actions()
        self.num_states = mdp.num_states()
        self.reset()
    
    def exec_sync(self):
        temp_s_by_a = np.zeros((self.num_states, self.num_actions))
        for a in range(self.num_actions):
            temp_s_by_a[:, a] = self.r[a] - self.r_bar + np.dot(self.P[a], self.v) - self.v
        delta = np.max(temp_s_by_a, axis=1)
        self.v += self.alpha * delta
        self.r_bar += self.beta * np.sum(delta)
        return
    
    def exec_async(self):
        idx = self.idx
        temp_a = np.zeros(self.num_actions)
        for a in range(self.num_actions):
            temp_a[a] = self.r[a][idx] - self.r_bar + np.dot(self.P[a][idx], self.v) - self.v[idx]
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
        # assert type(mrp) is MRP
        self.P = mrp.P.copy()
        self.r = mrp.r.copy()
        self.init_v = init_v.copy()
        self.init_r_bar = init_r_bar.copy()
        self.alpha = alpha
        self.beta = beta
        self.num_states = mrp.num_states()
        self.idx = 0
        self.reset()
    
    def exec_sync(self):
        self.r_bar = np.dot(self.P, self.r_bar)
        delta = self.r - self.r_bar + np.dot(self.P, self.v) - self.v
        self.v += self.alpha * delta
        self.r_bar += self.beta * delta
        return
    
    def exec_async(self):
        idx = self.idx
        self.r_bar[idx] = np.dot(self.P[idx], self.r_bar)
        delta = self.r[idx] - self.r_bar[idx] + np.dot(self.P[idx], self.v) - self.v[idx]
        self.v[idx] += self.alpha * delta
        self.r_bar[idx] += self.beta * delta
        self.idx = (self.idx + 1) % self.num_states
        return
    
    def reset(self):
        self.v = self.init_v.copy()
        self.r_bar = self.init_r_bar.copy()


class MDVI_Control1(object):
    def __init__(self, mdp, init_v, init_r_bar, alpha=1.0, beta=1.0, threshold=0.01):
        # assert type(mdp) is MDP
        self.P = mdp.P.copy()
        self.r = mdp.r.copy()
        self.init_v = init_v.copy()
        self.init_r_bar = init_r_bar.copy()
        self.alpha = alpha
        self.beta = beta
        self.num_states = mdp.num_states()
        self.num_actions = mdp.num_actions()
        self.idx = 0
        self.threshold = threshold
        self.reset()
    
    def exec_sync(self):
        num_states = self.num_states
        num_actions = self.num_actions
        temp_s_by_a = np.zeros((num_states, num_actions))
        for a in range(num_actions):
            temp_s_by_a[:, a] = np.dot(self.P[a], self.r_bar)
        self.r_bar = np.max(temp_s_by_a, axis=1)
        delta = np.zeros(num_states)
        for s in range(num_states):
            max_actions = np.where(temp_s_by_a[s] > self.r_bar[s] - self.threshold)[0]
            temp_a = np.zeros(max_actions.__len__())
            for i in range(max_actions.__len__()):
                temp_a[i] = self.r[max_actions[i]][s] - self.r_bar[s] + np.dot(self.P[max_actions[i]][s], self.v) - self.v[s]
            delta[s] = np.max(temp_a)
        self.v += self.alpha * delta
        self.r_bar += self.beta * delta
        return
    
    def exec_async(self):
        idx = self.idx
        num_actions = self.num_actions
        temp_a = np.zeros(num_actions)
        for a in range(num_actions):
            temp_a[a] = np.dot(self.P[a][idx], self.r_bar)
        self.r_bar[idx] = np.max(temp_a)
        max_actions = np.where(temp_a > self.r_bar[idx] - self.threshold)[0]
        temp_a = np.zeros(max_actions.__len__())
        for i in range(max_actions.__len__()):
            temp_a[i] = self.r[max_actions[i]][idx] - self.r_bar[idx] + np.dot(self.P[max_actions[i]][idx], self.v) - self.v[idx]
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
        # assert type(mdp) is MDP
        self.P = mdp.P.copy()
        self.r = mdp.r.copy()
        self.init_v = init_v.copy()
        self.init_r_bar = init_r_bar.copy()
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
            temp_s_by_a[:, a] = np.dot(self.P[a], self.r_bar)
        self.r_bar = np.max(temp_s_by_a, axis=1)
        temp_s_by_a = np.zeros((num_states, num_actions))
        for a in range(num_actions):
            temp_s_by_a[:, a] = self.r[a] - self.r_bar + np.dot(self.P[a], self.v) - self.v
        delta = np.max(temp_s_by_a, axis=1)
        self.v += self.alpha * delta
        self.r_bar += self.beta * delta
        return
    
    def exec_async(self):
        idx = self.idx
        num_actions = self.num_actions
        temp_a = np.zeros(num_actions)
        for a in range(num_actions):
            temp_a[a] = np.dot(self.P[a][idx], self.r_bar)
        self.r_bar[idx] = np.max(temp_a)
        temp_a = np.zeros(num_actions)
        for a in range(num_actions):
            temp_a[a] = self.r[a][idx] - self.r_bar[idx] + np.dot(self.P[a][idx], self.v) - self.v[idx]
        delta = np.max(temp_a)
        self.v[idx] += self.alpha * delta
        self.r_bar[idx] += self.beta * delta
        self.idx = (self.idx + 1) % self.num_states
        return
    
    def reset(self):
        self.v = self.init_v.copy()
        self.r_bar = self.init_r_bar.copy()