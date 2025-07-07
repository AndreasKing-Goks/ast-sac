## IMPORT FUNCTIONS
import ast_sac.torch.utils.pytorch_util as ptu

from ast_sac.data_management.env_replay_buffer import EnvReplayBuffer
from ast_sac.launchers.launcher_utils import setup_logger
from ast_sac.samplers.data_collector.path_collector import MdpPathCollector
from ast_sac.samplers.data_collector.rollout_functions import ast_sac_rollout
from ast_sac.torch.sac.policies.gaussian_policy import TanhGaussianPolicy, MakeDeterministic
from ast_sac.torch.sac.sac import SACTrainer
from ast_sac.torch.networks.mlp import ConcatMlp
from ast_sac.torch.core.torch_rl_algorithm import TorchBatchRLAlgorithm
from ast_sac.env_wrapper.normalized_box_env import NormalizedBoxEnv

def get_path_collector(env):
    ## Wrapped RL_Env with to accept normalized action from the policy
    expl_env = NormalizedBoxEnv(env)
    eval_env = NormalizedBoxEnv(env)

    ## Get the observation and action dimension
    obsv_dim = expl_env.observation_space.low.size
    action_dim = expl_env.action_space.low.size

    ## Prepare all the networks required for SAC
    M = 256

    ## Prepare the policy
    # For exploration
    policy = TanhGaussianPolicy(
        obs_dim=obsv_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )
    ## For evaluation (make deterministic)
    eval_policy = MakeDeterministic(policy)

    ## Prepare path collector for replay buffer
    # For evaluation
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
        rollout_fn=ast_sac_rollout
    )
    # For exploration
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
        rollout_fn=ast_sac_rollout
    )

    ## Prepare the replay buffer
    replay_buffer = EnvReplayBuffer(
        int(1E6),
        expl_env,
    )
    
    policy.to(ptu.device)
    eval_policy.to(ptu.device)
    
    return expl_env, eval_env, policy, eval_policy, eval_path_collector, expl_path_collector, replay_buffer