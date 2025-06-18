from collections import OrderedDict, namedtuple
from typing import Tuple

import numpy as np
import torch
import torch.optim as optim
from ast_sac.core.loss import LossFunction, LossStatistic
from torch import nn

import ast_sac.torch.utils.pytorch_util as ptu
from ast_sac.core.eval_util import create_stats_ordered_dict
# from ast_sac.torch.torch_rl_algorithm import TorchTrainer
from ast_sac.core.logging import add_prefix
import gtimer as gt