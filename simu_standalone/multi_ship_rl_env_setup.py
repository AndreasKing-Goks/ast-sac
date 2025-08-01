# import ast_sac
# print("ast_sac:", ast_sac)
# print("ast_sac.__file__:", getattr(ast_sac, "__file__", "No __file__"))

### IMPORT SIMULATOR ENVIRONMENTS
from rl_env.ship_in_transit.env import MultiShipRLEnv, ShipAssets

from rl_env.ship_in_transit.sub_systems.ship_model import  ShipConfiguration, EnvironmentConfiguration, SimulationConfiguration, ShipModelAST
from rl_env.ship_in_transit.sub_systems.ship_engine import MachinerySystemConfiguration, MachineryMode, MachineryModeParams, MachineryModes, SpecificFuelConsumptionBaudouin6M26Dot3, SpecificFuelConsumptionWartila6L26
from rl_env.ship_in_transit.sub_systems.LOS_guidance import LosParameters
from rl_env.ship_in_transit.sub_systems.obstacle import StaticObstacle, PolygonObstacle
from rl_env.ship_in_transit.sub_systems.controllers import ThrottleControllerGains, HeadingControllerGains, EngineThrottleFromSpeedSetPoint, HeadingBySampledRouteController
from rl_env.ship_in_transit.utils.print_termination import print_termination

## IMPORT FUNCTIONS
import ast_sac.torch.utils.pytorch_util as ptu

from ast_sac.data_management.env_replay_buffer import EnvReplayBuffer
from ast_sac.launchers.launcher_utils import setup_logger
from ast_sac.samplers.data_collector.path_collector import MdpPathCollector
from ast_sac.torch.sac.policies.gaussian_policy import TanhGaussianPolicy, MakeDeterministic
from ast_sac.torch.sac.sac import SACTrainer
from ast_sac.torch.networks.mlp import ConcatMlp
from ast_sac.torch.core.torch_rl_algorithm import TorchBatchRLAlgorithm
from ast_sac.env_wrapper.normalized_box_env import NormalizedBoxEnv
from ast_sac.samplers.data_collector.rollout_functions import rollout, ast_sac_rollout
from utils.animate import ShipTrajectoryAnimator, RLShipTrajectoryAnimator
from utils.paths_utils import get_data_path
from utils.center_plot import center_plot_window

from test_beds.path_reader import ActionPathReader
from test_beds.show_post_train_env_result import ShowPostTrainedEnvResult

### IMPORT TOOLS
import argparse
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
import torch
import time
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Argument Parser
parser = argparse.ArgumentParser(description='Ship Transit Soft Actor-Critic Args')

## Add arguments for environments
parser.add_argument('--max_sampling_frequency', type=int, default=9, metavar='N_SAMPLE',
                    help='ENV: maximum amount of action sampling per episode (default: 9)')
parser.add_argument('--time_step', type=int, default=4, metavar='TIMESTEP',
                    help='ENV: time step size in second for ship transit simulator (default: 2)')
parser.add_argument('--radius_of_acceptance', type=int, default=300, metavar='ROA',
                    help='ENV: radius of acceptance for LOS algorithm (default: 300)')
parser.add_argument('--lookahead_distance', type=int, default=1000, metavar='LD',
                    help='ENV: lookahead distance for LOS algorithm (default: 1000)')
parser.add_argument('--collav_mode', type=str, default='none', metavar='COLLAV_MODE',
                    help='ENV: collision avoidance mode. Modes are ["none", "simple", "sbmpc"] (default: "sbmpc")'),
parser.add_argument('--ship_draw', type=bool, default=True, metavar='SHIP_DRAW',
                    help='ENV: record ship drawing for plotting and animation (default: True)')
parser.add_argument('--time_since_last_ship_drawing', default=30, metavar='SHIP_DRAW_TIME',
                    help='ENV: time delay in second between ship drawing record (default: 30)')
parser.add_argument('--normalize_action', type=bool, default=False, metavar='NORM_ACT',
                    help='ENV: normalize environment action space (default: False)')

## Add arguments for soft actor-critic algorithm
parser.add_argument('--do_logging', type=bool, default=False, metavar='DO_LOG',
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
parser.add_argument('--num_epochs', type=int, default=3000, metavar='N_EPOCHS',
                    help='SAC_A: number of full training iterations (default: 3000)')
parser.add_argument('--num_eval_steps_per_epoch', type=int, default=180, metavar='N_EVAL_STEPS',
                    help='SAC_A: number of evaluation steps at the end of each epoch (default: 5000)') ## NEED TO CHECK
parser.add_argument('--num_trains_per_train_loop', type=int, default=360, metavar='N_TRAINS',
                    help='SAC_A: number of gradient updates to run per training loop (default: 1000)')
parser.add_argument('--num_expl_steps_per_train_loop', type=int, default=90, metavar='N_EXPL_STEPS',
                    help='SAC_A: number of exploration steps during training (default: 1000)')  ## NEED TO CHECK
parser.add_argument('--min_num_steps_before_training', type=int, default=270, metavar='MIN_N_STEPS',
                    help='SAC_A: delayed start — buffer pre-filled with random actions \
                                    to stabilize early learning (default: 1000)') # NEED TO CHECK
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

args = parser.parse_args()

# Engine configuration
main_engine_capacity = 2160e3
diesel_gen_capacity = 510e3
hybrid_shaft_gen_as_generator = 'GEN'
hybrid_shaft_gen_as_motor = 'MOTOR'
hybrid_shaft_gen_as_offline = 'OFF'

# Configure the simulation
ship_config = ShipConfiguration(
    coefficient_of_deadweight_to_displacement=0.7,
    bunkers=200000,
    ballast=200000,
    length_of_ship=80,
    width_of_ship=16,
    added_mass_coefficient_in_surge=0.4,
    added_mass_coefficient_in_sway=0.4,
    added_mass_coefficient_in_yaw=0.4,
    dead_weight_tonnage=3850000,
    mass_over_linear_friction_coefficient_in_surge=130,
    mass_over_linear_friction_coefficient_in_sway=18,
    mass_over_linear_friction_coefficient_in_yaw=90,
    nonlinear_friction_coefficient__in_surge=2400,
    nonlinear_friction_coefficient__in_sway=4000,
    nonlinear_friction_coefficient__in_yaw=400
)
env_config = EnvironmentConfiguration(
    current_velocity_component_from_north=-1,
    current_velocity_component_from_east=-1,
    wind_speed=2,
    wind_direction=-np.pi/4
)
pto_mode_params = MachineryModeParams(
    main_engine_capacity=main_engine_capacity,
    electrical_capacity=0,
    shaft_generator_state=hybrid_shaft_gen_as_generator
)
pto_mode = MachineryMode(params=pto_mode_params)

pti_mode_params = MachineryModeParams(
    main_engine_capacity=0,
    electrical_capacity=2*diesel_gen_capacity,
    shaft_generator_state=hybrid_shaft_gen_as_motor
)
pti_mode = MachineryMode(params=pti_mode_params)

mec_mode_params = MachineryModeParams(
    main_engine_capacity=main_engine_capacity,
    electrical_capacity=diesel_gen_capacity,
    shaft_generator_state=hybrid_shaft_gen_as_offline
)
mec_mode = MachineryMode(params=mec_mode_params)
mso_modes = MachineryModes(
    [pti_mode]
)
fuel_spec_me = SpecificFuelConsumptionWartila6L26()
fuel_spec_dg = SpecificFuelConsumptionBaudouin6M26Dot3()
machinery_config = MachinerySystemConfiguration(
    machinery_modes=mso_modes,
    machinery_operating_mode=0,
    linear_friction_main_engine=68,
    linear_friction_hybrid_shaft_generator=57,
    gear_ratio_between_main_engine_and_propeller=0.6,
    gear_ratio_between_hybrid_shaft_generator_and_propeller=0.6,
    propeller_inertia=6000,
    propeller_diameter=3.1,
    propeller_speed_to_torque_coefficient=7.5,
    propeller_speed_to_thrust_force_coefficient=1.7,
    hotel_load=200000,
    rated_speed_main_engine_rpm=1000,
    rudder_angle_to_sway_force_coefficient=50e3,
    rudder_angle_to_yaw_force_coefficient=500e3,
    max_rudder_angle_degrees=30,
    specific_fuel_consumption_coefficients_me=fuel_spec_me.fuel_consumption_coefficients(),
    specific_fuel_consumption_coefficients_dg=fuel_spec_dg.fuel_consumption_coefficients()
)

## CONFIGURE THE SHIP SIMULATION MODELS
# Ship in Test
ship_in_test_simu_setup = SimulationConfiguration(
    initial_north_position_m=100,
    initial_east_position_m=100,
    initial_yaw_angle_rad=60 * np.pi / 180,
    initial_forward_speed_m_per_s=4.25,
    initial_sideways_speed_m_per_s=0,
    initial_yaw_rate_rad_per_s=0,
    integration_step=args.time_step,
    simulation_time=10000,
)
test_initial_propeller_shaft_speed = 420
test_ship = ShipModelAST(ship_config=ship_config,
                       machinery_config=machinery_config,
                       environment_config=env_config,
                       simulation_config=ship_in_test_simu_setup,
                       initial_propeller_shaft_speed_rad_per_s=test_initial_propeller_shaft_speed * np.pi / 30)

# Obstacle Ship
ship_as_obstacle_simu_setup = SimulationConfiguration(
    initial_north_position_m=9900,
    initial_east_position_m=14900,
    initial_yaw_angle_rad=-135 * np.pi / 180,
    initial_forward_speed_m_per_s=3.5,
    initial_sideways_speed_m_per_s=0,
    initial_yaw_rate_rad_per_s=0,
    integration_step=args.time_step,
    simulation_time=10000,
)
obs_initial_propeller_shaft_speed = 200
obs_ship = ShipModelAST(ship_config=ship_config,
                       machinery_config=machinery_config,
                       environment_config=env_config,
                       simulation_config=ship_as_obstacle_simu_setup,
                       initial_propeller_shaft_speed_rad_per_s=obs_initial_propeller_shaft_speed * np.pi / 30)

## Configure the map data
map_data = [
    [(0,10000), (10000,10000), (9200,9000) , (7600,8500), (6700,7300), (4900,6500), (4300, 5400), (4700, 4500), (6000,4000), (5800,3600), (4200, 3200), (3200,4100), (2000,4500), (1000,4000), (900,3500), (500,2600), (0,2350)],   # Island 1 
    [(10000, 0), (11500,750), (12000, 2000), (11700, 3000), (11000, 3600), (11250, 4250), (12300, 4000), (13000, 3800), (14000, 3000), (14500, 2300), (15000, 1700), (16000, 800), (17500,0)], # Island 2
    [(15500, 10000), (16000, 9000), (18000, 8000), (19000, 7500), (20000, 6000), (20000, 10000)],
    [(5500, 5300), (6000,5000), (6800, 4500), (8000, 5000), (8700, 5500), (9200, 6700), (8000, 7000), (6700, 6300), (6000, 6000)],
    [(15000, 5000), (14000, 5500), (12500, 5000), (14000, 4100), (16000, 2000), (15700, 3700)],
    [(11000, 2000), (10300, 3200), (9000, 1500), (10000, 1000)]
    ]

map = PolygonObstacle(map_data)

## Set the throttle and autopilot controllers for the test ship
test_ship_throttle_controller_gains = ThrottleControllerGains(
    kp_ship_speed=205.25, ki_ship_speed=0.0525, kp_shaft_speed=50, ki_shaft_speed=0.00025
)
test_ship_throttle_controller = EngineThrottleFromSpeedSetPoint(
    gains=test_ship_throttle_controller_gains,
    max_shaft_speed=test_ship.ship_machinery_model.shaft_speed_max,
    time_step=args.time_step,
    initial_shaft_speed_integral_error=114
)

test_route_filename = 'test_ship_route.txt'
test_route_name = get_data_path(test_route_filename)
test_heading_controller_gains = HeadingControllerGains(kp=1.65, kd=75, ki=0.001)
test_los_guidance_parameters = LosParameters(
    radius_of_acceptance=args.radius_of_acceptance,
    lookahead_distance=args.lookahead_distance,
    integral_gain=0.002,
    integrator_windup_limit=4000
)
test_auto_pilot = HeadingBySampledRouteController(
    test_route_name,
    heading_controller_gains=test_heading_controller_gains,
    los_parameters=test_los_guidance_parameters,
    time_step=args.time_step,
    max_rudder_angle=machinery_config.max_rudder_angle_degrees * np.pi/180,
    num_of_samplings=2
)
test_desired_forward_speed =4.5

test_integrator_term = []
test_times = []

## Set the throttle and autopilot controllers for the obstacle ship
obs_ship_throttle_controller_gains = ThrottleControllerGains(
    kp_ship_speed=205.25, ki_ship_speed=0.0525, kp_shaft_speed=50, ki_shaft_speed=0.00025
)
obs_ship_throttle_controller = EngineThrottleFromSpeedSetPoint(
    gains=obs_ship_throttle_controller_gains,
    max_shaft_speed=obs_ship.ship_machinery_model.shaft_speed_max,
    time_step=args.time_step,
    initial_shaft_speed_integral_error=114
)

obs_route_filename = 'obs_ship_route.txt'
obs_route_name = get_data_path(obs_route_filename)
obs_heading_controller_gains = HeadingControllerGains(kp=1.65, kd=75, ki=0.001)
obs_los_guidance_parameters = LosParameters(
    radius_of_acceptance=args.radius_of_acceptance,
    lookahead_distance=args.lookahead_distance,
    integral_gain=0.002,
    integrator_windup_limit=4000
)
obs_auto_pilot = HeadingBySampledRouteController(
    obs_route_name,
    heading_controller_gains=obs_heading_controller_gains,
    los_parameters=obs_los_guidance_parameters,
    time_step=args.time_step,
    max_rudder_angle=machinery_config.max_rudder_angle_degrees * np.pi/180,
    num_of_samplings=2
)
obs_desired_forward_speed = 4.0 # 5.5 immediate near collision.

obs_integrator_term = []
obs_times = []

# Wraps simulation objects based on the ship type using a dictionary
test = ShipAssets(
    ship_model=test_ship,
    throttle_controller=test_ship_throttle_controller,
    auto_pilot=test_auto_pilot,
    desired_forward_speed=test_desired_forward_speed,
    integrator_term=test_integrator_term,
    time_list=test_times,
    stop_flag=False,
    type_tag='test_ship'
)

obs = ShipAssets(
    ship_model=obs_ship,
    throttle_controller=obs_ship_throttle_controller,
    auto_pilot=obs_auto_pilot,
    desired_forward_speed=obs_desired_forward_speed,
    integrator_term=obs_integrator_term,
    time_list=obs_times,
    stop_flag=False,
    type_tag='obs_ship'
)

# Package the assets for reinforcement learning agent
assets: List[ShipAssets] = [test, obs]

# Timer for drawing the ship
ship_draw = True
time_since_last_ship_drawing = 30

################################### ENV SPACE ###################################

# Initiate Multi-Ship Reinforcement Learning Environment Class Wrapper
env = MultiShipRLEnv(assets=assets,
                     map=map,
                     args=args)

expl_env = NormalizedBoxEnv(env)
eval_env = NormalizedBoxEnv(env)
obsv_dim = expl_env.observation_space.low.size
action_dim = expl_env.action_space.low.size

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
log = False
# log = True
if log:
    setup_logger('sanity_checks', variant=variant)
ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
M = variant['layer_size']

qf1 = ConcatMlp(
        input_size=obsv_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
qf2 = ConcatMlp(  
                
        input_size=obsv_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
target_qf1 = ConcatMlp(
        input_size=obsv_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
target_qf2 = ConcatMlp(
        input_size=obsv_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
policy = TanhGaussianPolicy(
        obs_dim=obsv_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )
eval_policy = MakeDeterministic(policy)
eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
        rollout_fn=ast_sac_rollout
    )
expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
        rollout_fn=ast_sac_rollout
    )
replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
algorithm.to(ptu.device)

print('-------------------------------------------------')

# Check normalized_box_env() performance
test1 = True
test1 = False

if test1:
    # Check normalized box env
    action_space_lo = env.action_space.low
    print('Basic env:', action_space_lo)
    action_space_norm_lo = expl_env.action_space.low
    print('Normalized box env:', action_space_norm_lo)
    print('-------------------------------------------------')

# Check action space sampled directly from environment
test2 = True
test2 = False

if test2:
    # Check the sampling
    action = env.action_space.sample()
    print('Basic env:', action)
    action_norm = expl_env.action_space.sample()
    print('Normalized box env:', action_norm)
    print('-------------------------------------------------')

# Check get_intermediate_waypoints() function behaviour
test3 = True
test3 = False

if test3:
    # Check for getting intermediate waypoints based on sampled action
    # Do it 4 times
    intermediate_waypoint_list = []

    print('sampling count before reset:', expl_env.wrapped_env.sampling_count)
    
    action = [np.deg2rad(0)]
    env.sampling_count += 1
    print('sampling count after first step:', expl_env.wrapped_env.sampling_count)
    print('action 1 in degree:', action[0])
    print('scoping angle 1 in degree:', np.rad2deg(action[0]))
    intermediate_waypoint = env.get_intermediate_waypoints(action)
    intermediate_waypoint_list.append(intermediate_waypoint)
    print('IW1:', intermediate_waypoint)

    action = [np.deg2rad(0)]
    env.sampling_count += 1
    print('sampling count after second step:', expl_env.wrapped_env.sampling_count)
    print('action 2 in degree:', action[0])
    print('scoping angle 2 in degree:', np.rad2deg(action[0]))
    intermediate_waypoint = env.get_intermediate_waypoints(action)
    intermediate_waypoint_list.append(intermediate_waypoint)
    print('IW2:', intermediate_waypoint)

    action = [np.deg2rad(0)]
    env.sampling_count += 1
    print('sampling count after third step:', expl_env.wrapped_env.sampling_count)
    print('action 3 in degree:', action[0])
    print('scoping angle 3 in degree:', np.rad2deg(action[0]))
    intermediate_waypoint = env.get_intermediate_waypoints(action)
    intermediate_waypoint_list.append(intermediate_waypoint)
    print('IW3:', intermediate_waypoint)

    action = [np.deg2rad(0)]
    env.sampling_count += 1
    print('sampling count after fourth step:', expl_env.wrapped_env.sampling_count)
    print('action 4 in degree:', action[0])
    print('scoping angle 4 in degree:', np.rad2deg(action[0]))
    intermediate_waypoint = env.get_intermediate_waypoints(action)
    intermediate_waypoint_list.append(intermediate_waypoint)
    print('IW4:', intermediate_waypoint)

    # Plot the points
    obs_route_n_start = obs.auto_pilot.navigate.north[0]
    obs_route_e_start = obs.auto_pilot.navigate.east[0]
    obs_route_n_end = obs.auto_pilot.navigate.north[-1]
    obs_route_e_end = obs.auto_pilot.navigate.east[-1]
    obs_route_end = [np.float64(obs_route_n_end), np.float64(obs_route_e_end)]
    obs_route_start = [np.float64(obs_route_n_start), np.float64(obs_route_e_start)]

    intermediate_waypoint_list.insert(0, obs_route_start)
    intermediate_waypoint_list.append(obs_route_end)

    north_list, east_list = zip(*intermediate_waypoint_list)

    # Create the figure
    plot=False
    plot=True
    if plot:
        plt.figure(figsize=(8, 6))
        plt.scatter(east_list, north_list, color='red', label='Sampled Waypoints')
        plt.plot(east_list, north_list, linestyle='--', color='gray', label='Waypoint Path')

        # Annotate each point with its index
        for idx, (e, n) in enumerate(zip(east_list, north_list)):
            plt.text(e, n, f'{idx}', fontsize=10, ha='right', va='bottom')

        # Label and style
        map.plot_obstacle(plt.gca())  # get current Axes to pass into map function
        plt.xlim(0, 20000)
        plt.ylim(0, 10000)
        plt.xlabel('East position (m)')
        plt.ylabel('North position (m)')
        plt.title('Intermediate Waypoints with Indices')
        plt.gca().set_aspect('equal')
        plt.grid(color='0.8', linestyle='-', linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()
    print('-------------------------------------------------')

# Test policy action sampling
test4 = True
test4 = False

if test4:
    # Move the policy model to the correct device (e.g., GPU if available)
    # This ensures all model parameters are on the same device as the input tensor
    policy.to(ptu.device)

    # Sample a random observation from the environment’s observation space
    # This gives a NumPy array sampled uniformly from the valid bounds
    sampled_obsv = expl_env.wrapped_env.observation_space.sample()

    # Display the sampled observation tensor
    print('Sampled observation from normalized env: \n', sampled_obsv)

    # Use the policy to sample an action based on the observation
    # The policy expects NumPy input, so we convert the tensor back to NumPy
    action, _ = policy.get_action(sampled_obsv)

    # Print the sampled action (output of the policy)
    print('Sampled action using policy:', action)
    print('-------------------------------------------------')

# Test the use of policy action sampling to step up the simulator
test5 = True
test5 = False

if test5:
    # Try resetting the environment with and without action
    print('Sampling count before old reset:', expl_env.wrapped_env.sampling_count)
    o = expl_env.reset()
    print('Sampling count after old reset :', expl_env.wrapped_env.sampling_count)
    o_waypoint_north = expl_env.obs.auto_pilot.navigate.north
    o_waypoint_east = expl_env.obs.auto_pilot.navigate.east

    # Reset to get initial states, sample an action using initial states, then reinitiate again using sampled action
    owa = expl_env.reset()                          # First reset
    policy.to(ptu.device)                           # THIS IS IMPORTANT
    action, _ = policy.get_action(owa)              # Get an action
    print('Sampling count before new reset:', expl_env.wrapped_env.sampling_count)
    owa = expl_env.reset(action)                    # Second reset  
    print('Sampling count after new reset :', expl_env.wrapped_env.sampling_count)
    owa_waypoint_north = expl_env.obs.auto_pilot.navigate.north
    owa_waypoint_east = expl_env.obs.auto_pilot.navigate.east

    print('Check if imediate route sampling during the init reset works')
    print('Initial states w/o action  :\n', o)
    print('North waypoints w/o action :', o_waypoint_north)
    print('East waypoints w/o action  :', o_waypoint_east)
    print('###')
    print('Initial states w/ action   :\n', owa)
    print('North waypoints w/ action  :', owa_waypoint_north)
    print('East waypoints w/ action   :', owa_waypoint_east)
    print('-------------------------------------------------')

# Test manual step() function with action sampled by a policy

test6 = True
test6 = False

if test6:
    # First we get an action
    init_obs = expl_env.reset()                 # Get initial observation
    policy.to(ptu.device)                       # Sent the policy to the device
    action, _ = policy.get_action(init_obs)     # Then sample initial action (normalized action [-1,1])
    print('Sampled action:', action)
    
    # Reset the environment
    print('Print reset, step up, and see how the simulation results is stored')
    expl_env.reset(action)
    print('init north:', expl_env.obs.ship_model.north)
    print('init east :', expl_env.obs.ship_model.east)
    print('init yaw_a:', expl_env.obs.ship_model.yaw_angle)
    print('init time :', expl_env.obs.ship_model.int.time)
    print('init north list:', expl_env.obs.ship_model.simulation_results['north position [m]'])
    print('init east list :', expl_env.obs.ship_model.simulation_results['east position [m]'])
    print('init yaw_a list:', expl_env.obs.ship_model.simulation_results['yaw angle [deg]'])
    print('init time list :', expl_env.obs.ship_model.simulation_results['time [s]'])
    print('init route north:', expl_env.obs.auto_pilot.navigate.north)
    print('init route east :', expl_env.obs.auto_pilot.navigate.east)
    next_observations, accumulated_reward, combined_done, env_info = expl_env.step(action)
    print('after step north:', expl_env.obs.ship_model.north)
    print('after step east :', expl_env.obs.ship_model.east)
    print('after step yaw_a:', expl_env.obs.ship_model.yaw_angle)
    print('after step time :', expl_env.obs.ship_model.int.time)
    print('after step north list:', expl_env.obs.ship_model.simulation_results['north position [m]'])
    print('after step east list :', expl_env.obs.ship_model.simulation_results['east position [m]'])
    print('after step yaw_a list:', expl_env.obs.ship_model.simulation_results['yaw angle [deg]'])
    print('after step time list :', expl_env.obs.ship_model.simulation_results['time [s]'])
    print('after step route north:', expl_env.obs.auto_pilot.navigate.north)
    print('after step route east :', expl_env.obs.auto_pilot.navigate.east)

    print('-------------------------------------------------')
test7 = True
test7 = False

if test7:
    
    # Manual scoping angle or random policy sampling
    manual = True
    # manual = False
    
    print('Test and plot step up behaviour')
    print('sampling count before reset:', expl_env.wrapped_env.sampling_count)
    init_observations = expl_env.reset()                                                    # First reset
    policy.to(ptu.device)                                                                   # Sent policy networks to device
    
    action, _ = policy.get_action(init_observations)                                        # Get an action
    if manual:
        action = env.do_normalize_action(np.deg2rad(0))
        print('First action', np.rad2deg(env.do_denormalize_action(action)))
        next_observations, accumulated_reward, combined_done, env_info = expl_env.step(action)  # Step up
        print('sampling count after first step:', expl_env.wrapped_env.sampling_count)  
        print('First step reward :', accumulated_reward)
    
    if not combined_done:
        action, _ = policy.get_action(next_observations)                                        # Get an action
        if manual:
            action = env.do_normalize_action(np.deg2rad(0))
        print('Second action', np.rad2deg(env.do_denormalize_action(action)))
        next_observations, accumulated_reward, combined_done, env_info = expl_env.step(action)  # Step up
        print('sampling count after second step:', expl_env.wrapped_env.sampling_count)
        print('Second step reward :', accumulated_reward)
    
    if not combined_done:
        action, _ = policy.get_action(next_observations)                                        # Get an action
        if manual:
            action = env.do_normalize_action(np.deg2rad(0))
        print('Third action', np.rad2deg(env.do_denormalize_action(action)))
        next_observations, accumulated_reward, combined_done, env_info = expl_env.step(action)  # Step upd
        print('sampling count after third step:', expl_env.wrapped_env.sampling_count)
        print('Third step reward :', accumulated_reward)
    
    if not combined_done:
        action, _ = policy.get_action(next_observations)                                        # Get an action
        if manual:            
            action = env.do_normalize_action(np.deg2rad(0))
        print('Fourth action', np.rad2deg(env.do_denormalize_action(action)))
        next_observations, accumulated_reward, combined_done, env_info = expl_env.step(action)  # Step up
        print('sampling count after fourth step:', expl_env.wrapped_env.sampling_count)
        print('Fourth step reward :', accumulated_reward)
    
    if not combined_done:
        action, _ = policy.get_action(next_observations)                                        # Get an action
        if manual:            
            action = env.do_normalize_action(np.deg2rad(0))
        print('Fourth action', np.rad2deg(env.do_denormalize_action(action)))
        next_observations, accumulated_reward, combined_done, env_info = expl_env.step(action)  # Step up
        print('sampling count after fifth step:', expl_env.wrapped_env.sampling_count)
        print('Fifth step reward :', accumulated_reward)
        
    if not combined_done:
        action, _ = policy.get_action(next_observations)                                        # Get an action
        if manual:            
            action = env.do_normalize_action(np.deg2rad(0))
        print('Fourth action', np.rad2deg(env.do_denormalize_action(action)))
        next_observations, accumulated_reward, combined_done, env_info = expl_env.step(action)  # Step up
        print('sampling count after sixth step:', expl_env.wrapped_env.sampling_count)
        print('Sixth step reward :', accumulated_reward)
    
    if not combined_done:
        action, _ = policy.get_action(next_observations)                                        # Get an action
        if manual:            
            action = env.do_normalize_action(np.deg2rad(0))
        print('Fourth action', np.rad2deg(env.do_denormalize_action(action)))
        next_observations, accumulated_reward, combined_done, env_info = expl_env.step(action)  # Step up
        print('sampling count after seventh step:', expl_env.wrapped_env.sampling_count)
        print('Seventh step reward :', accumulated_reward)
    
    if not combined_done:
        action, _ = policy.get_action(next_observations)                                        # Get an action
        if manual:            
            action = env.do_normalize_action(np.deg2rad(0))
        print('Fourth action', np.rad2deg(env.do_denormalize_action(action)))
        next_observations, accumulated_reward, combined_done, env_info = expl_env.step(action)  # Step up
        print('sampling count after eighth step:', expl_env.wrapped_env.sampling_count)
        print('Eigth step reward :', accumulated_reward)
    
    if not combined_done:
        action, _ = policy.get_action(next_observations)                                        # Get an action
        if manual:            
            action = env.do_normalize_action(np.deg2rad(0))
        print('Fourth action', np.rad2deg(env.do_denormalize_action(action)))
        next_observations, accumulated_reward, combined_done, env_info = expl_env.step(action)  # Step up
        print('sampling count after ninth step:', expl_env.wrapped_env.sampling_count)
        print('Ninth step reward :', accumulated_reward)

    # Get the simulation results for all assets
    ts_results_df = pd.DataFrame().from_dict(expl_env.wrapped_env.test.ship_model.simulation_results)
    os_results_df = pd.DataFrame().from_dict(expl_env.wrapped_env.obs.ship_model.simulation_results)

    # Get the time when the sampling begin, failure modes and the waypoints
    waypoint_sampling_times = expl_env.wrapped_env.waypoint_sampling_times
    
    # Print
    print('')
    print('---')
    print('Waypoint sampling time record:', waypoint_sampling_times)
    print('---')
    print('after step route north:', expl_env.obs.auto_pilot.navigate.north)
    print('after step route east :', expl_env.obs.auto_pilot.navigate.east)
    print('---')
    print(env_info['events'])

    # Plot 1: Overall process plot
    plot_1 = False
    # plot_1 = True

    # Plot 2: Status plot
    plot_2 = False
    # plot_2 = True
    
    # Plot 3: Reward plots
    plot_3 = False
    # plot_3 = True
    
    # Plot 4: Cumulative rewards plots
    plot_4 = False
    plot_4 = True
    
    # For animation
    animation = False
    animation = True

    if animation:
        test_route = {'east': test.auto_pilot.navigate.east, 'north': test.auto_pilot.navigate.north}
        obs_route = {'east': obs.auto_pilot.navigate.east, 'north': obs.auto_pilot.navigate.north}
        waypoint_sampling_times = expl_env.wrapped_env.waypoint_sampling_times
        waypoints = list(zip(obs.auto_pilot.navigate.east, obs.auto_pilot.navigate.north))
        map_obj = map
        radius_of_acceptance = args.radius_of_acceptance
        timestamps = test.time_list
        interval = args.time_step * 1000
        test_drawer=test.ship_model.draw
        obs_drawer=obs.ship_model.draw
        test_headings=ts_results_df["yaw angle [deg]"]
        obs_headings=os_results_df["yaw angle [deg]"]
        is_collision_imminent_list = expl_env.wrapped_env.is_collision_imminent_list
        is_collision_list = expl_env.wrapped_env.is_collision_list
        
        # Do animation
        animator = RLShipTrajectoryAnimator(ts_results_df,
                                            os_results_df,
                                            test_route,
                                            obs_route,
                                            waypoint_sampling_times,
                                            waypoints,
                                            map_obj,
                                            radius_of_acceptance,
                                            timestamps,
                                            interval,
                                            test_drawer,
                                            obs_drawer,
                                            test_headings,
                                            obs_headings,
                                            is_collision_imminent_list,
                                            is_collision_list)

        ani_ID = 1
        ani_dir = os.path.join('animation', 'comparisons')
        filename  = "trajectory.mp4"
        video_path = os.path.join(ani_dir, filename)
        fps = 480
        
        # Create the output directory if it doesn't exist
        os.makedirs(ani_dir, exist_ok=True)
        
        # animator.save(video_path, fps)
        animator.run(fps)
        
    # Create a No.4 2x4 grid for accumulated reward subplots
    if plot_4:
        fig_4, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 6))
        plt.figure(fig_4.number)
        axes = axes.flatten()
        
        # Center plotting
        center_plot_window()

        titles = [
            'Ship under test grounding rewards (cumulative)',
            'Ship under test navigational failure rewards (cumulative)',
            'Rewards from ship under test (cumulative)',
            'Ship collision rewards (cumulative)',
            'Obstacle ship grounding rewards (cumulative)',
            'Obstacle ship navigational failure rewards (cumulative)',
            'Rewards from obstacle ship (cumulative)',
            'Total rewards obtained (cumulative)'
        ]

        reward_series = [
            expl_env.wrapped_env.reward_tracker.test_ship_grounding,
            expl_env.wrapped_env.reward_tracker.test_ship_nav_failure,
            expl_env.wrapped_env.reward_tracker.from_test_ship,
            expl_env.wrapped_env.reward_tracker.ship_collision,
            expl_env.wrapped_env.reward_tracker.obs_ship_grounding,
            expl_env.wrapped_env.reward_tracker.obs_ship_nav_failure,
            expl_env.wrapped_env.reward_tracker.from_obs_ship,
            expl_env.wrapped_env.reward_tracker.total,
        ]

        for i in range(8):
            cumulative_reward = np.cumsum(reward_series[i])
            axes[i].plot(cumulative_reward)
            axes[i].set_title(titles[i], fontsize=10)
            axes[i].set_xlabel('Time (s)', fontsize=8)
            axes[i].set_ylabel('Cumulative Rewards', fontsize=8)
            axes[i].tick_params(axis='both', labelsize=7)
            axes[i].grid(color='0.8', linestyle='-', linewidth=0.3)
            axes[i].set_xlim(left=0)

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.3, hspace=0.4)

    
    # Create a No.3 2x4 grid for subplots
    if plot_3:
        fig_3, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 6))  # Slightly shorter height
        plt.figure(fig_3.number)
        axes = axes.flatten()

        # Center plotting
        center_plot_window()
        
        titles = [
            'Ship under test grounding rewards',
            'Ship under test navigational failure rewards',
            'Rewards from ship under test',
            'Ship collision rewards',
            'Obstacle ship grounding rewards',
            'Obstacle ship navigational failure rewards',
            'Rewards from obstacle ship',
            'Total rewards obtained'
        ]

        reward_series = [
            expl_env.wrapped_env.reward_tracker.test_ship_grounding,
            expl_env.wrapped_env.reward_tracker.test_ship_nav_failure,
            expl_env.wrapped_env.reward_tracker.from_test_ship,
            expl_env.wrapped_env.reward_tracker.ship_collision,
            expl_env.wrapped_env.reward_tracker.obs_ship_grounding,
            expl_env.wrapped_env.reward_tracker.obs_ship_nav_failure,
            expl_env.wrapped_env.reward_tracker.from_obs_ship,
            expl_env.wrapped_env.reward_tracker.total,
        ]

        for i in range(8):
            axes[i].plot(reward_series[i])
            axes[i].set_title(titles[i], fontsize=10)
            axes[i].set_xlabel('Time (s)', fontsize=8)
            axes[i].set_ylabel('Rewards', fontsize=8)
            axes[i].tick_params(axis='both', labelsize=7)
            axes[i].grid(color='0.8', linestyle='-', linewidth=0.3)
            axes[i].set_xlim(left=0)

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.3, hspace=0.4)  # Optional: fine-tune spacing

    # Create a No.2 3x4 grid for subplots
    if plot_2:
        fig_2, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10))
        plt.figure(fig_2.number)  # Ensure it's the current figure
        axes = axes.flatten()  # Flatten the 2D array for easier indexing
    
        # Center plotting
        center_plot_window()

        # Plot 2.1: Forward Speed
        axes[0].plot(ts_results_df['time [s]'], ts_results_df['forward speed [m/s]'])
        axes[0].axhline(y=test_desired_forward_speed, color='red', linestyle='--', linewidth=1.5, label='Desired Forward Speed')
        axes[0].set_title('Test Ship Forward Speed [m/s]')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Forward Speed (m/s)')
        axes[0].grid(color='0.8', linestyle='-', linewidth=0.5)
        axes[0].set_xlim(left=0)

        # Plot 3.1: Forward Speed
        axes[1].plot(os_results_df['time [s]'], os_results_df['forward speed [m/s]'])
        axes[1].axhline(y=obs_desired_forward_speed, color='red', linestyle='--', linewidth=1.5, label='Desired Forward Speed')
        axes[1].set_title('Obstacle Ship Forward Speed [m/s]')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Forward Speed (m/s)')
        axes[1].grid(color='0.8', linestyle='-', linewidth=0.5)
        axes[1].set_xlim(left=0)

        # Plot 2.2: Rudder Angle
        axes[2].plot(ts_results_df['time [s]'], ts_results_df['rudder angle [deg]'])
        axes[2].set_title('Test Ship Rudder angle [deg]')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Rudder angle [deg]')
        axes[2].grid(color='0.8', linestyle='-', linewidth=0.5)
        axes[2].set_xlim(left=0)
        axes[2].set_ylim(-31,31)

        # Plot 3.2: Rudder Angle
        axes[3].plot(os_results_df['time [s]'], os_results_df['rudder angle [deg]'])
        axes[3].set_title('Obstacle Ship Rudder angle [deg]')
        axes[3].set_xlabel('Time (s)')
        axes[3].set_ylabel('Rudder angle [deg]')
        axes[3].grid(color='0.8', linestyle='-', linewidth=0.5)
        axes[3].set_xlim(left=0)
        axes[3].set_ylim(-31,31)

        # Plot 2.3: Cross Track error
        axes[4].plot(ts_results_df['time [s]'], ts_results_df['cross track error [m]'])
        axes[4].set_title('Test Ship Cross Track Error [m]')
        axes[4].set_xlabel('Time (s)')
        axes[4].set_ylabel('Cross track error (m)')
        axes[4].grid(color='0.8', linestyle='-', linewidth=0.5)
        axes[4].set_xlim(left=0)

        # Plot 3.3: Cross Track error
        axes[5].plot(os_results_df['time [s]'], os_results_df['cross track error [m]'])
        axes[5].set_title('Obstacle Ship Cross Track Error [m]')
        axes[5].set_xlabel('Time (s)')
        axes[5].set_ylabel('Cross track error (m)')
        axes[5].grid(color='0.8', linestyle='-', linewidth=0.5)
        axes[5].set_xlim(left=0)

        # Plot 2.4: Propeller Shaft Speed
        axes[6].plot(ts_results_df['time [s]'], ts_results_df['propeller shaft speed [rpm]'])
        axes[6].set_title('Test Ship Propeller Shaft Speed [rpm]')
        axes[6].set_xlabel('Time (s)')
        axes[6].set_ylabel('Propeller Shaft Speed (rpm)')
        axes[6].grid(color='0.8', linestyle='-', linewidth=0.5)
        axes[6].set_xlim(left=0)

        # Plot 3.4: Propeller Shaft Speed
        axes[7].plot(os_results_df['time [s]'], os_results_df['propeller shaft speed [rpm]'])
        axes[7].set_title('Obstacle Ship Propeller Shaft Speed [rpm]')
        axes[7].set_xlabel('Time (s)')
        axes[7].set_ylabel('Propeller Shaft Speed (rpm)')
        axes[7].grid(color='0.8', linestyle='-', linewidth=0.5)
        axes[7].set_xlim(left=0)

        # Plot 2.5: Power vs Available Power
        axes[8].plot(ts_results_df['time [s]'], ts_results_df['power electrical [kw]'], label="Power")
        axes[8].plot(ts_results_df['time [s]'], ts_results_df['available power electrical [kw]'], label="Available Power")
        axes[8].set_title('Test Ship Power vs Available Power [kw]')
        axes[8].set_xlabel('Time (s)')
        axes[8].set_ylabel('Power (kw)')
        axes[8].legend()
        axes[8].grid(color='0.8', linestyle='-', linewidth=0.5)
        axes[8].set_xlim(left=0)

        # Plot 3.5: Power vs Available Power
        axes[9].plot(os_results_df['time [s]'], os_results_df['power electrical [kw]'], label="Power")
        axes[9].plot(os_results_df['time [s]'], os_results_df['available power electrical [kw]'], label="Available Power")
        axes[9].set_title('Obstacle Ship Power vs Available Power [kw]')
        axes[9].set_xlabel('Time (s)')
        axes[9].set_ylabel('Power (kw)')
        axes[9].legend()
        axes[9].grid(color='0.8', linestyle='-', linewidth=0.5)
        axes[9].set_xlim(left=0)

        # Plot 2.6: Fuel Consumption
        axes[10].plot(ts_results_df['time [s]'], ts_results_df['fuel consumption [kg]'])
        axes[10].set_title('Test Ship Fuel Consumption [kg]')
        axes[10].set_xlabel('Time (s)')
        axes[10].set_ylabel('Fuel Consumption (kg)')
        axes[10].grid(color='0.8', linestyle='-', linewidth=0.5)
        axes[10].set_xlim(left=0)

        # Plot 3.6: Fuel Consumption
        axes[11].plot(os_results_df['time [s]'], os_results_df['fuel consumption [kg]'])
        axes[11].set_title('Obstacle Ship Fuel Consumption [kg]')
        axes[11].set_xlabel('Time (s)')
        axes[11].set_ylabel('Fuel Consumption (kg)')
        axes[11].grid(color='0.8', linestyle='-', linewidth=0.5)
        axes[11].set_xlim(left=0)

        # Adjust layout for better spacing
        plt.tight_layout()

    if plot_1:    
        plt.figure(figsize=(10, 5.5))

        # Plot 1.1: Ship trajectory with sampled route
        # Test ship
        plt.plot(ts_results_df['east position [m]'].to_numpy(), ts_results_df['north position [m]'].to_numpy())
        plt.scatter(test.auto_pilot.navigate.east, test.auto_pilot.navigate.north, marker='x', color='blue')  # Waypoints
        plt.plot(test.auto_pilot.navigate.east, test.auto_pilot.navigate.north, linestyle='--', color='blue')  # Line
        for x, y in zip(test.ship_model.ship_drawings[1], test.ship_model.ship_drawings[0]):
            plt.plot(x, y, color='blue')
        # for i, (east, north) in enumerate(zip(test.auto_pilot.navigate.east, test.auto_pilot.navigate.north)):
        #     test_radius_circle = Circle((east, north), args.radius_of_acceptance, color='blue', alpha=0.3, fill=True)
        #     plt.gca().add_patch(test_radius_circle)
        # Obs ship    
        plt.plot(os_results_df['east position [m]'].to_numpy(), os_results_df['north position [m]'].to_numpy())
        plt.scatter(obs.auto_pilot.navigate.east, obs.auto_pilot.navigate.north, marker='x', color='red')  # Waypoints
        plt.plot(obs.auto_pilot.navigate.east, obs.auto_pilot.navigate.north, linestyle='--', color='red')  # Line
        for x, y in zip(obs.ship_model.ship_drawings[1], obs.ship_model.ship_drawings[0]):
            plt.plot(x, y, color='red')
        for i, (east, north) in enumerate(zip(obs.auto_pilot.navigate.east, obs.auto_pilot.navigate.north)):
            obs_radius_circle = Circle((east, north), args.radius_of_acceptance, color='red', alpha=0.3, fill=True)
            plt.gca().add_patch(obs_radius_circle)
        map.plot_obstacle(plt.gca())  # get current Axes to pass into map function

        plt.xlim(0, 20000)
        plt.ylim(0, 10000)
        plt.title('Ship Trajectory with the Sampled Route')
        plt.xlabel('East position (m)')
        plt.ylabel('North position (m)')
        plt.gca().set_aspect('equal')
        plt.grid(color='0.8', linestyle='-', linewidth=0.5)

        # Adjust layout for better spacing
        plt.tight_layout()
    
    # Show Plot
    plt.show()

    print('-------------------------------------------------')

# Test step - reset -step again
test8 = True
test8 = False

if test8:
    # Manual scoping angle or random policy sampling
    manual = True
    manual = False
    
    print('Test and plot step up behaviour')
    print('sampling count before reset:', expl_env.wrapped_env.sampling_count)
    init_observations = expl_env.reset()                                                    # First reset
    policy.to(ptu.device)                                                                   # Sent policy networks to device
    
    action, _ = policy.get_action(init_observations)                                        # Get an action
    if manual:
        action = env.do_normalize_action(np.deg2rad(0))
    print('First action', np.rad2deg(env.do_denormalize_action(action)))
    next_observations, accumulated_reward, combined_done, env_info = expl_env.step(action)  # Step up
    print('sampling count after first step:', expl_env.wrapped_env.sampling_count)  
    
    if not combined_done:
        action, _ = policy.get_action(next_observations)                                        # Get an action
        if manual:
            action = env.do_normalize_action(np.deg2rad(0))
        print('Second action', np.rad2deg(env.do_denormalize_action(action)))
        next_observations, accumulated_reward, combined_done, env_info = expl_env.step(action)  # Step up
        print('sampling count after second step:', expl_env.wrapped_env.sampling_count)
    
    if not combined_done:
        action, _ = policy.get_action(next_observations)                                        # Get an action
        if manual:
            action = env.do_normalize_action(np.deg2rad(0))
        print('Third action', np.rad2deg(env.do_denormalize_action(action)))
        next_observations, accumulated_reward, combined_done, env_info = expl_env.step(action)  # Step upd
        print('sampling count after third step:', expl_env.wrapped_env.sampling_count)
    
    if not combined_done:
        action, _ = policy.get_action(next_observations)                                        # Get an action
        if manual:            
            action = env.do_normalize_action(np.deg2rad(0))
        print('Fourth action', np.rad2deg(env.do_denormalize_action(action)))
        next_observations, accumulated_reward, combined_done, env_info = expl_env.step(action)  # Step up
        print('sampling count after fourth step:', expl_env.wrapped_env.sampling_count)
        
    ### THEN RESET
        # Manual scoping angle or random policy sampling
    # manual = True
    # manual = False
    
    print('############################################################################')
    print('DO RESET')
    print('############################################################################')
    
    print('Test and plot step up behaviour')
    print('sampling count before reset:', expl_env.wrapped_env.sampling_count)
    init_observations = expl_env.reset()                                                    # First reset
    policy.to(ptu.device)                                                                   # Sent policy networks to device
    
    action, _ = policy.get_action(init_observations)                                        # Get an action
    if manual:
        action = env.do_normalize_action(np.deg2rad(0))
    print('First action', np.rad2deg(env.do_denormalize_action(action)))
    next_observations, accumulated_reward, combined_done, env_info = expl_env.step(action)  # Step up
    print('sampling count after first step:', expl_env.wrapped_env.sampling_count)  
    
    if not combined_done:
        action, _ = policy.get_action(next_observations)                                        # Get an action
        if manual:
            action = env.do_normalize_action(np.deg2rad(0))
        print('Second action', np.rad2deg(env.do_denormalize_action(action)))
        next_observations, accumulated_reward, combined_done, env_info = expl_env.step(action)  # Step up
        print('sampling count after second step:', expl_env.wrapped_env.sampling_count)
    
    if not combined_done:
        action, _ = policy.get_action(next_observations)                                        # Get an action
        if manual:
            action = env.do_normalize_action(np.deg2rad(0))
        print('Third action', np.rad2deg(env.do_denormalize_action(action)))
        next_observations, accumulated_reward, combined_done, env_info = expl_env.step(action)  # Step upd
        print('sampling count after third step:', expl_env.wrapped_env.sampling_count)
    
    if not combined_done:
        action, _ = policy.get_action(next_observations)                                        # Get an action
        if manual:            
            action = env.do_normalize_action(np.deg2rad(0))
        print('Fourth action', np.rad2deg(env.do_denormalize_action(action)))
        next_observations, accumulated_reward, combined_done, env_info = expl_env.step(action)  # Step up
        print('sampling count after fourth step:', expl_env.wrapped_env.sampling_count)

    # Get the simulation results for all assets
    ts_results_df = pd.DataFrame().from_dict(expl_env.wrapped_env.test.ship_model.simulation_results)
    os_results_df = pd.DataFrame().from_dict(expl_env.wrapped_env.obs.ship_model.simulation_results)

    # Get the time when the sampling begin, failure modes and the waypoints
    waypoint_sampling_times = expl_env.wrapped_env.waypoint_sampling_times
    
    # Print
    print('')
    print('---')
    print('Waypoint sampling time record:', waypoint_sampling_times)
    print('---')
    print('after step route north:', expl_env.obs.auto_pilot.navigate.north)
    print('after step route east :', expl_env.obs.auto_pilot.navigate.east)
    print('---')
    print(env_info['events'])

    # Plot 1: Overall process plot
    plot_1 = False
    # plot_1 = True

    # Plot 2: Status plot
    plot_2 = False
    # plot_2 = True
    
    # Plot 3: Reward plots
    plot_3 = False
    # plot_3 = True
    
    # Plot 4: Cumulative rewards plots
    plot_4 = False
    # plot_4 = True
    
    # For animation
    animation = False
    animation = True

    if animation:
        test_route = {'east': test.auto_pilot.navigate.east, 'north': test.auto_pilot.navigate.north}
        obs_route = {'east': obs.auto_pilot.navigate.east, 'north': obs.auto_pilot.navigate.north}
        waypoint_sampling_times = expl_env.wrapped_env.waypoint_sampling_times
        waypoints = list(zip(obs.auto_pilot.navigate.east, obs.auto_pilot.navigate.north))
        map_obj = map
        radius_of_acceptance = args.radius_of_acceptance
        timestamps = test.time_list
        interval = args.time_step * 1000
        test_drawer=test.ship_model.draw
        obs_drawer=obs.ship_model.draw
        test_headings=ts_results_df["yaw angle [deg]"]
        obs_headings=os_results_df["yaw angle [deg]"]
        is_collision_imminent_list = expl_env.wrapped_env.is_collision_imminent_list
        is_collision_list = expl_env.wrapped_env.is_collision_list
        
        # Do animation
        animator = RLShipTrajectoryAnimator(ts_results_df,
                                            os_results_df,
                                            test_route,
                                            obs_route,
                                            waypoint_sampling_times,
                                            waypoints,
                                            map_obj,
                                            radius_of_acceptance,
                                            timestamps,
                                            interval,
                                            test_drawer,
                                            obs_drawer,
                                            test_headings,
                                            obs_headings,
                                            is_collision_imminent_list,
                                            is_collision_list)

        ani_ID = 1
        ani_dir = os.path.join('animation', 'comparisons')
        filename  = "trajectory.mp4"
        video_path = os.path.join(ani_dir, filename)
        fps = 480
        
        # Create the output directory if it doesn't exist
        os.makedirs(ani_dir, exist_ok=True)
        
        # animator.save(video_path, fps)
        animator.run(fps)
        
    # Create a No.4 2x4 grid for accumulated reward subplots
    if plot_4:
        fig_4, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 6))
        plt.figure(fig_4.number)
        axes = axes.flatten()
        
        # Center plotting
        center_plot_window()

        titles = [
            'Ship under test grounding rewards (cumulative)',
            'Ship under test navigational failure rewards (cumulative)',
            'Rewards from ship under test (cumulative)',
            'Ship collision rewards (cumulative)',
            'Obstacle ship grounding rewards (cumulative)',
            'Obstacle ship navigational failure rewards (cumulative)',
            'Rewards from obstacle ship (cumulative)',
            'Total rewards obtained (cumulative)'
        ]

        reward_series = [
            expl_env.wrapped_env.reward_tracker.test_ship_grounding,
            expl_env.wrapped_env.reward_tracker.test_ship_nav_failure,
            expl_env.wrapped_env.reward_tracker.from_test_ship,
            expl_env.wrapped_env.reward_tracker.ship_collision,
            expl_env.wrapped_env.reward_tracker.obs_ship_grounding,
            expl_env.wrapped_env.reward_tracker.obs_ship_nav_failure,
            expl_env.wrapped_env.reward_tracker.from_obs_ship,
            expl_env.wrapped_env.reward_tracker.total,
        ]

        for i in range(8):
            cumulative_reward = np.cumsum(reward_series[i])
            axes[i].plot(cumulative_reward)
            axes[i].set_title(titles[i], fontsize=10)
            axes[i].set_xlabel('Time (s)', fontsize=8)
            axes[i].set_ylabel('Cumulative Rewards', fontsize=8)
            axes[i].tick_params(axis='both', labelsize=7)
            axes[i].grid(color='0.8', linestyle='-', linewidth=0.3)
            axes[i].set_xlim(left=0)

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.3, hspace=0.4)

    
    # Create a No.3 2x4 grid for subplots
    if plot_3:
        fig_3, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 6))  # Slightly shorter height
        plt.figure(fig_3.number)
        axes = axes.flatten()

        # Center plotting
        center_plot_window()
        
        titles = [
            'Ship under test grounding rewards',
            'Ship under test navigational failure rewards',
            'Rewards from ship under test',
            'Ship collision rewards',
            'Obstacle ship grounding rewards',
            'Obstacle ship navigational failure rewards',
            'Rewards from obstacle ship',
            'Total rewards obtained'
        ]

        reward_series = [
            expl_env.wrapped_env.reward_tracker.test_ship_grounding,
            expl_env.wrapped_env.reward_tracker.test_ship_nav_failure,
            expl_env.wrapped_env.reward_tracker.from_test_ship,
            expl_env.wrapped_env.reward_tracker.ship_collision,
            expl_env.wrapped_env.reward_tracker.obs_ship_grounding,
            expl_env.wrapped_env.reward_tracker.obs_ship_nav_failure,
            expl_env.wrapped_env.reward_tracker.from_obs_ship,
            expl_env.wrapped_env.reward_tracker.total,
        ]

        for i in range(8):
            axes[i].plot(reward_series[i])
            axes[i].set_title(titles[i], fontsize=10)
            axes[i].set_xlabel('Time (s)', fontsize=8)
            axes[i].set_ylabel('Rewards', fontsize=8)
            axes[i].tick_params(axis='both', labelsize=7)
            axes[i].grid(color='0.8', linestyle='-', linewidth=0.3)
            axes[i].set_xlim(left=0)

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.3, hspace=0.4)  # Optional: fine-tune spacing

    # Create a No.2 3x4 grid for subplots
    if plot_2:
        fig_2, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10))
        plt.figure(fig_2.number)  # Ensure it's the current figure
        axes = axes.flatten()  # Flatten the 2D array for easier indexing
    
        # Center plotting
        center_plot_window()

        # Plot 2.1: Forward Speed
        axes[0].plot(ts_results_df['time [s]'], ts_results_df['forward speed [m/s]'])
        axes[0].axhline(y=test_desired_forward_speed, color='red', linestyle='--', linewidth=1.5, label='Desired Forward Speed')
        axes[0].set_title('Test Ship Forward Speed [m/s]')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Forward Speed (m/s)')
        axes[0].grid(color='0.8', linestyle='-', linewidth=0.5)
        axes[0].set_xlim(left=0)

        # Plot 3.1: Forward Speed
        axes[1].plot(os_results_df['time [s]'], os_results_df['forward speed [m/s]'])
        axes[1].axhline(y=obs_desired_forward_speed, color='red', linestyle='--', linewidth=1.5, label='Desired Forward Speed')
        axes[1].set_title('Obstacle Ship Forward Speed [m/s]')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Forward Speed (m/s)')
        axes[1].grid(color='0.8', linestyle='-', linewidth=0.5)
        axes[1].set_xlim(left=0)

        # Plot 2.2: Rudder Angle
        axes[2].plot(ts_results_df['time [s]'], ts_results_df['rudder angle [deg]'])
        axes[2].set_title('Test Ship Rudder angle [deg]')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Rudder angle [deg]')
        axes[2].grid(color='0.8', linestyle='-', linewidth=0.5)
        axes[2].set_xlim(left=0)
        axes[2].set_ylim(-31,31)

        # Plot 3.2: Rudder Angle
        axes[3].plot(os_results_df['time [s]'], os_results_df['rudder angle [deg]'])
        axes[3].set_title('Obstacle Ship Rudder angle [deg]')
        axes[3].set_xlabel('Time (s)')
        axes[3].set_ylabel('Rudder angle [deg]')
        axes[3].grid(color='0.8', linestyle='-', linewidth=0.5)
        axes[3].set_xlim(left=0)
        axes[3].set_ylim(-31,31)

        # Plot 2.3: Cross Track error
        axes[4].plot(ts_results_df['time [s]'], ts_results_df['cross track error [m]'])
        axes[4].set_title('Test Ship Cross Track Error [m]')
        axes[4].set_xlabel('Time (s)')
        axes[4].set_ylabel('Cross track error (m)')
        axes[4].grid(color='0.8', linestyle='-', linewidth=0.5)
        axes[4].set_xlim(left=0)

        # Plot 3.3: Cross Track error
        axes[5].plot(os_results_df['time [s]'], os_results_df['cross track error [m]'])
        axes[5].set_title('Obstacle Ship Cross Track Error [m]')
        axes[5].set_xlabel('Time (s)')
        axes[5].set_ylabel('Cross track error (m)')
        axes[5].grid(color='0.8', linestyle='-', linewidth=0.5)
        axes[5].set_xlim(left=0)

        # Plot 2.4: Propeller Shaft Speed
        axes[6].plot(ts_results_df['time [s]'], ts_results_df['propeller shaft speed [rpm]'])
        axes[6].set_title('Test Ship Propeller Shaft Speed [rpm]')
        axes[6].set_xlabel('Time (s)')
        axes[6].set_ylabel('Propeller Shaft Speed (rpm)')
        axes[6].grid(color='0.8', linestyle='-', linewidth=0.5)
        axes[6].set_xlim(left=0)

        # Plot 3.4: Propeller Shaft Speed
        axes[7].plot(os_results_df['time [s]'], os_results_df['propeller shaft speed [rpm]'])
        axes[7].set_title('Obstacle Ship Propeller Shaft Speed [rpm]')
        axes[7].set_xlabel('Time (s)')
        axes[7].set_ylabel('Propeller Shaft Speed (rpm)')
        axes[7].grid(color='0.8', linestyle='-', linewidth=0.5)
        axes[7].set_xlim(left=0)

        # Plot 2.5: Power vs Available Power
        axes[8].plot(ts_results_df['time [s]'], ts_results_df['power electrical [kw]'], label="Power")
        axes[8].plot(ts_results_df['time [s]'], ts_results_df['available power electrical [kw]'], label="Available Power")
        axes[8].set_title('Test Ship Power vs Available Power [kw]')
        axes[8].set_xlabel('Time (s)')
        axes[8].set_ylabel('Power (kw)')
        axes[8].legend()
        axes[8].grid(color='0.8', linestyle='-', linewidth=0.5)
        axes[8].set_xlim(left=0)

        # Plot 3.5: Power vs Available Power
        axes[9].plot(os_results_df['time [s]'], os_results_df['power electrical [kw]'], label="Power")
        axes[9].plot(os_results_df['time [s]'], os_results_df['available power electrical [kw]'], label="Available Power")
        axes[9].set_title('Obstacle Ship Power vs Available Power [kw]')
        axes[9].set_xlabel('Time (s)')
        axes[9].set_ylabel('Power (kw)')
        axes[9].legend()
        axes[9].grid(color='0.8', linestyle='-', linewidth=0.5)
        axes[9].set_xlim(left=0)

        # Plot 2.6: Fuel Consumption
        axes[10].plot(ts_results_df['time [s]'], ts_results_df['fuel consumption [kg]'])
        axes[10].set_title('Test Ship Fuel Consumption [kg]')
        axes[10].set_xlabel('Time (s)')
        axes[10].set_ylabel('Fuel Consumption (kg)')
        axes[10].grid(color='0.8', linestyle='-', linewidth=0.5)
        axes[10].set_xlim(left=0)

        # Plot 3.6: Fuel Consumption
        axes[11].plot(os_results_df['time [s]'], os_results_df['fuel consumption [kg]'])
        axes[11].set_title('Obstacle Ship Fuel Consumption [kg]')
        axes[11].set_xlabel('Time (s)')
        axes[11].set_ylabel('Fuel Consumption (kg)')
        axes[11].grid(color='0.8', linestyle='-', linewidth=0.5)
        axes[11].set_xlim(left=0)

        # Adjust layout for better spacing
        plt.tight_layout()

    if plot_1:    
        plt.figure(figsize=(10, 5.5))

        # Plot 1.1: Ship trajectory with sampled route
        # Test ship
        plt.plot(ts_results_df['east position [m]'].to_numpy(), ts_results_df['north position [m]'].to_numpy())
        plt.scatter(test.auto_pilot.navigate.east, test.auto_pilot.navigate.north, marker='x', color='blue')  # Waypoints
        plt.plot(test.auto_pilot.navigate.east, test.auto_pilot.navigate.north, linestyle='--', color='blue')  # Line
        for x, y in zip(test.ship_model.ship_drawings[1], test.ship_model.ship_drawings[0]):
            plt.plot(x, y, color='blue')
        # for i, (east, north) in enumerate(zip(test.auto_pilot.navigate.east, test.auto_pilot.navigate.north)):
        #     test_radius_circle = Circle((east, north), args.radius_of_acceptance, color='blue', alpha=0.3, fill=True)
        #     plt.gca().add_patch(test_radius_circle)
        # Obs ship    
        plt.plot(os_results_df['east position [m]'].to_numpy(), os_results_df['north position [m]'].to_numpy())
        plt.scatter(obs.auto_pilot.navigate.east, obs.auto_pilot.navigate.north, marker='x', color='red')  # Waypoints
        plt.plot(obs.auto_pilot.navigate.east, obs.auto_pilot.navigate.north, linestyle='--', color='red')  # Line
        for x, y in zip(obs.ship_model.ship_drawings[1], obs.ship_model.ship_drawings[0]):
            plt.plot(x, y, color='red')
        for i, (east, north) in enumerate(zip(obs.auto_pilot.navigate.east, obs.auto_pilot.navigate.north)):
            obs_radius_circle = Circle((east, north), args.radius_of_acceptance, color='red', alpha=0.3, fill=True)
            plt.gca().add_patch(obs_radius_circle)
        map.plot_obstacle(plt.gca())  # get current Axes to pass into map function

        plt.xlim(0, 20000)
        plt.ylim(0, 10000)
        plt.title('Ship Trajectory with the Sampled Route')
        plt.xlabel('East position (m)')
        plt.ylabel('North position (m)')
        plt.gca().set_aspect('equal')
        plt.grid(color='0.8', linestyle='-', linewidth=0.5)

        # Adjust layout for better spacing
        plt.tight_layout()
    
    # Show Plot
    plt.show()

    print('-------------------------------------------------')

# Test ast_sac_rollout() alone
test9=True
test9=False

if test9:
    start_time = time.time()
    policy.to(ptu.device)   
    # Get path using the function
    path = ast_sac_rollout(expl_env,
                           policy,
                           max_path_length=args.max_sampling_frequency)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Elasped time:', elapsed_time, 'seconds')
    
    for i in range(len(path['actions'])):
        print('STEP', i+1)
        print('Observation      :', path['observations'][i])
        print('Action           :', path['actions'][i])
        print('Next observation :', path['next_observations'][i])
        print('Rewards          :', path['rewards'][i])
        print('Terminal         :', path['terminals'][i])
        print('Done             :', path['dones'][i])
        # print('Env infos        :', path['env_infos'][i])
        print('------------------')

    print('Sum Reward       :', np.sum(path['rewards']))
    print('Tracked Reward   :', expl_env.wrapped_env.accumulated_rewards_list)
    
    # Do plot and animate from the post trained environment
    post_trained = ShowPostTrainedEnvResult(expl_env)
    post_trained.do_plot_and_animate(
                                    #  plot_1=True,
                                    #  plot_2=True,
                                     plot_3=True,
                                    #  plot_4=True,
                                     animation=True
                                     )
    
    print('#######################################################################')
    print('#######################################################################')
    print('#######################################################################')
    
    # # Get path using the function
    # path = ast_sac_rollout(expl_env,
    #                        policy,
    #                        max_path_length=args.max_sampling_frequency)
    
    # for i in range(len(path['actions'])):
    #     print('STEP', i+1)
    #     print('Observation      :', path['observations'][i])
    #     print('Action           :', path['actions'][i])
    #     print('Next observation :', path['next_observations'][i])
    #     print('Rewards          :', path['rewards'][i])
    #     print('Terminal         :', path['terminals'][i])
    #     print('Done             :', path['dones'][i])
    #     print('Env infos        :', path['env_infos'][i])
    #     print('------------------')
        
    # print('Sum Reward       :', np.sum(path['rewards']))
    # print('Tracked Reward   :', expl_env.wrapped_env.accumulated_rewards_list)
    
    # # Do plot and animate from the post trained environment
    # post_trained = ShowPostTrainedEnvResult(expl_env)
    # post_trained.do_plot_and_animate(
    #                                 #  plot_1=True,
    #                                 #  plot_2=True,
    #                                  plot_3=True,
    #                                 #  plot_4=True,
    #                                  animation=True
    #                                  )