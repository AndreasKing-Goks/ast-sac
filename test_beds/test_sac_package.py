# import ast_sac
# print("ast_sac:", ast_sac)
# print("ast_sac.__file__:", getattr(ast_sac, "__file__", "No __file__"))

### IMPORT SIMULATOR ENVIRONMENTS
from rl_env.ship_in_transit.env import MultiShipEnv, ShipAssets

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
from utils.basic_animate import ShipTrajectoryAnimator
from utils.paths_utils import get_data_path
from utils.center_plot import center_plot_window

### IMPORT TOOLS
import argparse
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
import torch
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

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
#                     help='Step for doing full action sampling (default:1000')
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
    simulation_time=None,
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
    simulation_time=None,
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
    [(15500, 10000), (16000, 9000), (18000, 8000), (19000, 7500), (20000, 6000), (20000, 10000)]
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
# Set Collav Mode
collav_mode = None
collav_mode = 'simple'
collav_mode = 'sbmpc'

# Initiate Multi-Ship Reinforcement Learning Environment Class Wrapper
env = MultiShipEnv(assets=assets,
                     map=map,
                     ship_draw=ship_draw,
                     collav=collav_mode,
                     time_since_last_ship_drawing=time_since_last_ship_drawing,
                     args=args)