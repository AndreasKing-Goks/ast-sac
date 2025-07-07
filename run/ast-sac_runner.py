## IMPORT ENV PREP SCRIPT
from run.env_setup import prepare_multiship_rl_env

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

## IMPORT TOOLS
import argparse


def parse_cli_args():
    ## Argument Parser
    parser = argparse.ArgumentParser(description='Ship Transit Soft Actor-Critic Args')

    ## Add arguments for environments
    parser.add_argument('--max_sampling_frequency', type=int, default=9, metavar='N_SAMPLE',
                        help='ENV: maximum amount of action sampling per episode (default: 9)')
    parser.add_argument('--time_step', type=int, default=2, metavar='TIMESTEP',
                        help='ENV: time step size in second for ship transit simulator (default: 2)')
    parser.add_argument('--radius_of_acceptance', type=int, default=250, metavar='ROA',
                        help='ENV: radius of acceptance for LOS algorithm (default: 250)')
    parser.add_argument('--lookahead_distance', type=int, default=1000, metavar='LD',
                        help='ENV: lookahead distance for LOS algorithm (default: 1000)')
    parser.add_argument('--collav_mode', type=str, default='sbmpc', metavar='COLLAV_MODE',
                        help='ENV: collision avoidance mode. Modes are ["none", "simple", "sbmpc"] (default: "sbmpc")'),
    parser.add_argument('--ship_draw', type=bool, default=True, metavar='SHIP_DRAW',
                        help='ENV: record ship drawing for plotting and animation (default: True)')
    parser.add_argument('--time_since_last_ship_drawing', default=30, metavar='SHIP_DRAW_TIME',
                        help='ENV: time delay in second between ship drawing record (default: 30)')
    parser.add_argument('--normalize_action', type=bool, default=False, metavar='NORM_ACT',
                        help='ENV: normalize environment action space (default: False)')
    
    ## Add arguments for soft actor-critic algorithm
    parser.add_argument('--do_logging', type=bool, default=True, metavar='DO_LOG',
                        help='SAC_A: Activate training logging (default: True)')
    parser.add_argument('--algorithm', type=str, default='SAC', metavar='RL_ALG',
                        help='SAC_A: RL algorithm type for AST (default: "SAC")')
    parser.add_argument('--version', type=str, default='normal', metavar='VERSION',
                        help='SAC_A: RL version (default: "normal")')
    parser.add_argument('--layer_size', type=int, default=256, metavar='LAYER_SIZE',
                        help='SAC_A: hidden layer size for all neural networks (default: 256)')
    parser.add_argument('--replay_buffer_size', type=int, default=300000, metavar='BUFFER_SIZE',
                        help='SAC_A: replay buffer size (default: 1E6)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='BATCH_SIZE',
                        help='SAC_A: data batch size for training (default: 256)')
    # EPOCHS/TRAINS/STEPS COUNT
    # step  : a single interaction with the environment
    #           - collection of (s, a, r, s'. done)
    # train : a single gradient update on the policy/networks
    #           - training step to get transition from replay buffer, compute losses, then backpropagate
    #           - a single batch size use is counted as a single training step
    # epoch  : a full training cycle of learning + evaluation + logging
    parser.add_argument('--num_epochs', type=int, default=30, metavar='N_EPOCHS',
                        help='SAC_A: number of full training iterations (default: 3000)')
    parser.add_argument('--num_eval_steps_per_epoch', type=int, default=30, metavar='N_EVAL_STEPS',
                        help='SAC_A: number of evaluation steps at the end of each epoch (default: 180)') ## NEED TO CHECK
    parser.add_argument('--num_trains_per_train_loop', type=int, default=30, metavar='N_TRAINS',
                        help='SAC_A: number of gradient updates to run per training loop (default: 360)')
    parser.add_argument('--num_expl_steps_per_train_loop', type=int, default=30, metavar='N_EXPL_STEPS',
                        help='SAC_A: number of exploration steps during training (default: 90)')  ## NEED TO CHECK
    parser.add_argument('--min_num_steps_before_training', type=int, default=30, metavar='MIN_N_STEPS',
                        help='SAC_A: delayed start — buffer pre-filled with random actions \
                                     to stabilize early learning (default: 270)') # NEED TO CHECK
    parser.add_argument('--max_path_length', type=int, default=9, metavar='MAX_PATH_LEN',
                        help='SAC_A: maximum number of steps per episode before termination (default: 9)') # NEED TO CHECK
    
    ## Add arguments for soft actor-critic trainer
    parser.add_argument('--discount', type=float, default=0.99, metavar='DISCOUNT_FACTOR',
                        help='SAC_T: discount factor for future rewards (default: 0.99)')
    parser.add_argument('--soft_target_tau', type=float, default=5E-3, metavar='SAC_TEMP',
                        help='SAC_T: temperature factor for target network soft updates (default: 5E-3)')
    parser.add_argument('--target_update_period', type=int, default=1, metavar='TARGET_UPDATE',
                        help='SAC_T: target network weights update counts in steps (default: 1)')
    parser.add_argument('--policy_lr', type=float, default=3E-4, metavar='POLICY_LR',
                        help='SAC_T: policy networks learning rate (default: 3E-4)')
    parser.add_argument('--qf_lr', type=float, default=3E-4, metavar='QF_LR',
                        help='SAC_T: Q-function networks learning rate (default: 3E-4)')
    parser.add_argument('--reward_scale', type=int, default=1, metavar='REWARD_SCALE',
                        help='SAC_T: scale factor for rewards (default: 1)')
    parser.add_argument('--use_automatic_entropy_tuning', type=bool, default=True, metavar='AUTO_ENTROPY',
                        help='SAC_T: adaptive entropy coefficient tuning if True (default: True)')
    
    ## Parse args
    args = parser.parse_args()
    
    return args


def experiment(variant, args):    
    ## Prepare RL_Env, use normalized env.
    multi_ship_rl_env, _ = prepare_multiship_rl_env(args)
    
    ## Wrapped RL_Env with to accept normalized action from the policy
    expl_env = NormalizedBoxEnv(multi_ship_rl_env, reward_scale=args.reward_scale)
    eval_env = NormalizedBoxEnv(multi_ship_rl_env, reward_scale=args.reward_scale)
    
    ## Get the observation and action dimension
    obsv_dim = expl_env.observation_space.low.size
    action_dim = expl_env.action_space.low.size

    ## Prepare all the networks required for SAC
    M = variant['layer_size']
    # Q-network 1
    qf1 = ConcatMlp(
        input_size=obsv_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    # Q-network 2
    qf2 = ConcatMlp(
        input_size=obsv_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    # Target Q-network 1
    target_qf1 = ConcatMlp(
        input_size=obsv_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    # Target Q-network 2
    target_qf2 = ConcatMlp(
        input_size=obsv_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    
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
        variant['replay_buffer_size'],
        expl_env,
    )
    
    ## Prepare the SAC Trainer for AST-SAC
    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    
    ## Prepare the algorithm to train the AST-SAC for batch RL training
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    
    ## Put all the network to predetermined computational device
    algorithm.to(ptu.device)
    
    ## Train the algorithm
    algorithm.train()


if __name__ == "__main__":

    # Get the args from args parser
    args = parse_cli_args()
    
    ## Perpare the RL variant
    variant = dict(
        algorithm=args.algorithm,
        version=args.version,
        layer_size=args.layer_size,
        replay_buffer_size=args.replay_buffer_size,
        algorithm_kwargs=dict(
            num_epochs=args.num_epochs,
            num_eval_steps_per_epoch=args.num_eval_steps_per_epoch,
            num_trains_per_train_loop=args.num_trains_per_train_loop,
            num_expl_steps_per_train_loop=args.num_expl_steps_per_train_loop,
            min_num_steps_before_training=args.min_num_steps_before_training,
            max_path_length=args.max_path_length,
            batch_size=args.batch_size,
        ),
        trainer_kwargs=dict(
            discount=args.discount,
            soft_target_tau=args.soft_target_tau,
            target_update_period=args.target_update_period,
            policy_lr=args.policy_lr,
            qf_lr=args.qf_lr,
            reward_scale=args.reward_scale,
            use_automatic_entropy_tuning=args.use_automatic_entropy_tuning,
        ),
    )
    
    ## Log the experiment 
    if args.do_logging:
        setup_logger('ast-sac_maritime_logs', variant=variant)
    
    ## Set the GPU available 
    ptu.set_gpu_mode(True)      # optionally set the GPU (default=False)
    
    ## Do AST-SAC training
    experiment(variant, args)