from widowx_envs.policies.policy import Policy
from widowx_envs.utils.utils import AttrDict
from widowx_envs.control_loops import Environment_Exception
import widowx_envs.utils.transformation_utils as tr

from pyquaternion import Quaternion
import numpy as np
import time

class TestLatency(Policy):
    def __init__(self, ag_params, policyparams):
        """ Computes actions from states/observations. """
        self._hp = self._default_hparams()
        self._override_defaults(policyparams)

    def _default_hparams(self):
        dict = AttrDict(
            type=None,
        )
        default_dict = super(Policy, self)._default_hparams()
        default_dict.update(dict)
        return default_dict


    def act(self, t):
        print('policy at t', t)
        if t == 4:
            action = {'actions': np.array([0.03, 0, 0, 1])}
        else:
            action = {'actions': np.array([0, 0, 0, 1])}
        print('action', action)
        return action


    def reset(self):
        pass
