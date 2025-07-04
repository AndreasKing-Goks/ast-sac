import argparse

def get_env_args():
    # Argument Parser
    parser = argparse.ArgumentParser(description='Ship Transit Soft Actor-Critic Args')

    parser.add_argument('--max_sampling_frequency', type=int, default=7, metavar='N_SAMPLE',
                        help='maximum amount of action sampling per episode (default: 7)')
    parser.add_argument('--time_step', type=int, default=2, metavar='TIMESTEP',
                        help='time step size in second for ship transit simulator (default: 2)')
    parser.add_argument('--radius_of_acceptance', type=int, default=200, metavar='ROA',
                        help='radius of acceptance for LOS algorithm (default: 200)')
    parser.add_argument('--lookahead_distance', type=int, default=1000, metavar='LD',
                        help='lookahead distance for LOS algorithm (default: 1000)')
    parser.add_argument('--collav_mode', type=str, default='sbmpc', metavar='COLLAV_MODE',
                        help='collision avoidance mode. Mode are [None, "simple", "sbmpc"] (default: "sbmpc")'),
    parser.add_argument('--ship_draw', type=bool, default=True, metavar='SHIP_DRAW',
                        help='record ship drawing for plotting and animation (default: True)')
    parser.add_argument('--time_since_last_ship_drawing', default=30, metavar='SHIP_DRAW_TIME',
                        help='time delay in second between ship drawing record (default: 30)')
    parser.add_argument('--normalize_action', type=bool, default=False, metavar='NORM_ACT',
                        help='normalize environment action space (default: False)')

    args = parser.parse_args()
    return args