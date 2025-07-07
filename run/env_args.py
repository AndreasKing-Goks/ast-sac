import argparse

def get_env_args():
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
    
    ## Parse args
    args = parser.parse_args()
    
    return args