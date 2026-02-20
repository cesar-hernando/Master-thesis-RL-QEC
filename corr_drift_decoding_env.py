'''
In this file, we build a Gymnasium environment for reweighting the decoding graph based on the edge correlations 
statistics and the first MWPM pass selected edges.
'''

import numpy as np
import stim
import pymatching
import gymnasium as gym

class DecodingGraphReweightingEnv(gym.Env):
    """
    A Gymnasium environment for reweighting the decoding graph based on the edge correlations
    and weights statistics as well as the first MWPM pass selected edges.
    """

    def __init__(self):
        super().__init__()

        # Define action and observation spaces (to be implemented)
        # self.action_space = ...
        # self.observation_space = ...

    def reset(self):
        pass