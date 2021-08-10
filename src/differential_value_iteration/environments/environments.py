# MRP with 3 states in a loop with a reward of 3 in one of the transitions
# (like in Exercise 10.7 of Sutton and Barto's (2018) textbook)

class ThreeLoopMRP():

    def __init__(self):

        self.num_states = 3
        self.R = [0,0,3]
        self.P = [
        [0,1,0],
        [0,0,1],
        [1,0,0]
        ]
