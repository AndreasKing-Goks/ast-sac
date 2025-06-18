'''
Policy that always outputs zero
'''

from ast_sac.policies.base import Policy

class RandomPolicy(Policy):
    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, obs):
        return self.action_space.sample(), {}