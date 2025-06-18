import abc
from collections import OrderedDict

from typing import Iterable
from torch import nn

# from ast_sac.core.batch_rl_algorithm import BatchRLAgorithm
# from ast_sac.core.online_rl_algorithm import OnlineRLAlgorithm
# from ast_sac.core.trainer import Trainer
from ast_sac.torch.core.module import np_to_pytorch_batch