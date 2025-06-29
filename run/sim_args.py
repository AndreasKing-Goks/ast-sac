import argparse

def get_sim_args():
    # Argument Parser
    parser = argparse.ArgumentParser(description='Ship Transit Soft Actor-Critic Args')

    # Coefficient and boolean parameters
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every scoring episode (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--theta', type=float, default=2, metavar='G',
                        help='action sampling frequency coefficient(θ) (default: 1.5)')
    parser.add_argument('--sampling_frequency', type=int, default=4, metavar='G',
                        help='maximum amount of action sampling per episode (default: 9)')
    parser.add_argument('--max_route_resampling', type=int, default=1000, metavar='G',
                        help='maximum amount of route resampling if route is sampled inside\
                        obstacle (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                              term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                        help='Automaically adjust α (default: False)')

    # Neural networks parameters
    parser.add_argument('--seed', type=int, default=25350, metavar='Q',
                        help='random seed (default: 123456)')
    parser.add_argument('--batch_size', type=int, default=64, metavar='Q',
                        help='batch size (default: 256)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='Q',
                        help='size of replay buffer (default: 1000000)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='Q',
                        help='hidden size (default: 256)')
    parser.add_argument('--cuda', action="store_true", default=True,
                        help='run on CUDA (default: False)')

    # Timesteps and episode parameters
    parser.add_argument('--time_step', type=int, default=2, metavar='N',
                        help='time step size in second for ship transit simulator (default: 2)')
    parser.add_argument('--num_steps', type=int, default=100, metavar='N',
                        help='maximum number of steps across all episodes (default: 100000)')
    parser.add_argument('--num_steps_episode', type=int, default=10000, metavar='N',
                        help='Maximum number of steps per episode to avoid infinite recursion (default: 1000)')
    parser.add_argument('--start_steps', type=int, default=1000, metavar='N',
                        help='SAC starting steps for sampling random actions (default: 1000)')
    parser.add_argument('--update_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    # parser.add_argument('--sampling_step', type=int, default=1000, metavar='N',
    #                    help='Step for doing full action sampling (default:1000')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--scoring_episode_every', type=int, default=50, metavar='N',
                        help='Number of every episode to evaluate learning performance(default: 40)')
    parser.add_argument('--num_scoring_episodes', type=int, default=25, metavar='N',
                        help='Number of episode for learning performance assesment(default: 20)')

    # Others
    parser.add_argument('--radius_of_acceptance', type=int, default=300, metavar='O',
                        help='Radius of acceptance for LOS algorithm(default: 600)')
    parser.add_argument('--lookahead_distance', type=int, default=1000, metavar='O',
                        help='Lookahead distance for LOS algorithm(default: 450)')
    parser.add_argument('--collav_mode', type=str, default='sbmpc', metavar='O',
                        help='Collision avoidance mode. Mode=None, "simple", "sbmpc" (default: "sbmpc")')

    args = parser.parse_args()
    return args