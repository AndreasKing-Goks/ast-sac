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
                    help='time step size in second for ship transit simulator (default: 0.5)')
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
    initial_yaw_angle_rad=45 * np.pi / 180,
    initial_forward_speed_m_per_s=0,
    initial_sideways_speed_m_per_s=0,
    initial_yaw_rate_rad_per_s=0,
    integration_step=args.time_step,
    simulation_time=None,
)
test_ship = ShipModelAST(ship_config=ship_config,
                       machinery_config=machinery_config,
                       environment_config=env_config,
                       simulation_config=ship_in_test_simu_setup,
                       initial_propeller_shaft_speed_rad_per_s=400 * np.pi / 30)

# Obstacle Ship
ship_as_obstacle_simu_setup = SimulationConfiguration(
    initial_north_position_m=9900,
    initial_east_position_m=14900,
    initial_yaw_angle_rad=-90 * np.pi / 180,
    initial_forward_speed_m_per_s=0,
    initial_sideways_speed_m_per_s=0,
    initial_yaw_rate_rad_per_s=0,
    integration_step=args.time_step,
    simulation_time=None,
)
obs_ship = ShipModelAST(ship_config=ship_config,
                       machinery_config=machinery_config,
                       environment_config=env_config,
                       simulation_config=ship_as_obstacle_simu_setup,
                       initial_propeller_shaft_speed_rad_per_s=400 * np.pi / 30)

## Configure the map data
map_data = [
    [(0,10000), (10000,10000), (9200,9000) , (7600,8500), (6700,7300), (4900,6500), (4300, 5400), (4700, 4500), (6000,4000), (5800,3600), (4200, 3200), (3200,4100), (2000,4500), (1000,4000), (900,3500), (500,2600), (0,2350)],   # Island 1 
    [(10000, 0), (11500,750), (12000, 2000), (11700, 3000), (11000, 3600), (11250, 4250), (12300, 4000), (13000, 3800), (14000, 3000), (14500, 2300), (15000, 1700), (16000, 800), (17500,0)], # Island 2
    [(15500, 10000), (16000, 9000), (18000, 8000), (19000, 7500), (20000, 6000), (20000, 10000)]
    ]

map = PolygonObstacle(map_data)

## Set the throttle and autopilot controllers for the test ship
test_ship_throttle_controller_gains = ThrottleControllerGains(
    kp_ship_speed=0.05, ki_ship_speed=0.13, kp_shaft_speed=0.1, ki_shaft_speed=0.005
)
test_ship_throttle_controller = EngineThrottleFromSpeedSetPoint(
    gains=test_ship_throttle_controller_gains,
    max_shaft_speed=test_ship.ship_machinery_model.shaft_speed_max,
    time_step=args.time_step,
    initial_shaft_speed_integral_error=114
)

test_route_filename = 'test_ship_route.txt'
test_route_name = get_data_path(test_route_filename)
test_heading_controller_gains = HeadingControllerGains(kp=0.9, kd=50, ki=0.00001)
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
test_desired_forward_speed = 5.0

test_integrator_term = []
test_times = []

## Set the throttle and autopilot controllers for the obstacle ship
obs_ship_throttle_controller_gains = ThrottleControllerGains(
    kp_ship_speed=7, ki_ship_speed=0.13, kp_shaft_speed=0.05, ki_shaft_speed=1
)
obs_ship_throttle_controller = EngineThrottleFromSpeedSetPoint(
    gains=test_ship_throttle_controller_gains,
    max_shaft_speed=obs_ship.ship_machinery_model.shaft_speed_max,
    time_step=args.time_step,
    initial_shaft_speed_integral_error=114
)

obs_route_filename = 'obs_ship_route.txt'
obs_route_name = get_data_path(obs_route_filename)
obs_heading_controller_gains = HeadingControllerGains(kp=0.9, kd=50, ki=0.00001)
obs_los_guidance_parameters = LosParameters(
    radius_of_acceptance=args.radius_of_acceptance,
    lookahead_distance=args.lookahead_distance,
    integral_gain=0.002,
    integrator_windup_limit=4000
)
obs_auto_pilot = HeadingBySampledRouteController(
    obs_route_name,
    heading_controller_gains=obs_heading_controller_gains,
    los_parameters=test_los_guidance_parameters,
    time_step=args.time_step,
    max_rudder_angle=machinery_config.max_rudder_angle_degrees * np.pi/180,
    num_of_samplings=2
)
obs_desired_forward_speed = 5.0 # 8.0

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
# collav_mode = 'sbmpc'

# Initiate Multi-Ship Reinforcement Learning Environment Class Wrapper
env = MultiShipEnv(assets=assets,
                     map=map,
                     ship_draw=ship_draw,
                     collav=collav_mode,
                     time_since_last_ship_drawing=time_since_last_ship_drawing,
                     args=args)

states = env.states

################################################################################################################################
##################################################### SIMULATION STARTS HERE ###################################################
################################################################################################################################
# Place the assets in the simulator by resetting the environment
env.reset()

# Simulator records
total_numsteps = 0

# Set done as False before simulation
done = False

# Run the simulator unitl it terminates, iteratively
while not done:
        
    ## STEP UP THE SIMULATION
    next_states, done, termination_cond = env.step()
    total_numsteps += 1
            
    # Set the next state as current state for the next simulator step
    states = next_states
################################################################################################################################
################################################################################################################################

# Print last status
print_termination(termination_cond)

ts_results_df = pd.DataFrame().from_dict(test.ship_model.simulation_results)
os_results_df = pd.DataFrame().from_dict(obs.ship_model.simulation_results)

# For animation
animation = False
# animation = True

if animation:
    test_route = {'east': test.auto_pilot.navigate.east, 'north': test.auto_pilot.navigate.north}
    obs_route = {'east': obs.auto_pilot.navigate.east, 'north': obs.auto_pilot.navigate.north}
    waypoint_reached_times = None
    waypoints = list(zip(obs.auto_pilot.navigate.east, obs.auto_pilot.navigate.north))
    map_obj = map
    radius_of_acceptance = args.radius_of_acceptance
    timestamps = test.time_list
    interval = args.time_step * 1000
    test_drawer=test.ship_model.draw
    obs_drawer=obs.ship_model.draw
    test_headings=ts_results_df["yaw angle [deg]"]
    obs_headings=os_results_df["yaw angle [deg]"]
    
    # Do animation
    animator = ShipTrajectoryAnimator(ts_results_df,
                                      os_results_df,
                                      test_route,
                                      obs_route,
                                      waypoint_reached_times,
                                      waypoints,
                                      map_obj,
                                      radius_of_acceptance,
                                      timestamps,
                                      interval,
                                      test_drawer,
                                      obs_drawer,
                                      test_headings,
                                      obs_headings)

    ani_ID = 1
    ani_dir = os.path.join('animation', 'sbmpc')
    filename  = "trajectory.mp4"
    video_path = os.path.join(ani_dir, filename)
    fps = 480
    
    # Create the output directory if it doesn't exist
    os.makedirs(ani_dir, exist_ok=True)
    
    # animator.save(video_path, fps)
    animator.run(fps)

## SHOW PLOT
# Plot 1: Map plot
plot_1 = False
# plot_1 = True

# Plot 2: Status plot
plot_2 = False
plot_2 = True

# Plot 3: Position plot
plot_3 = False
# plot_3 = True

# Create a plot 1 single figure and axis instead of a grid
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
    
# Create a No.2 3x4 grid for subplots
if plot_2:
    fig_2, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10))
    axes = axes.flatten()  # Flatten the 2D array for easier indexing

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

# Create a No.4 2x2 grid for subplots
if plot_3:
    fig_3, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
    axes = axes.flatten()  # Flatten the 2D array for easier indexing

    # Plot 4.1: Test north position
    axes[0].plot(ts_results_df['time [s]'], ts_results_df['north position [m]'])
    axes[0].set_title('Test Ship North Position [m]')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Position (m)')
    axes[0].grid(color='0.8', linestyle='-', linewidth=0.5)
    axes[0].set_xlim(left=0)

    # Plot 4.2: Obsacle north position
    axes[1].plot(os_results_df['time [s]'], os_results_df['north position [m]'])
    axes[1].set_title('Obstacle Ship North Position [m]')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Position (m)')
    axes[1].grid(color='0.8', linestyle='-', linewidth=0.5)
    axes[1].set_xlim(left=0)

    # Plot 4.3: Test east position
    axes[2].plot(ts_results_df['time [s]'], ts_results_df['east position [m]'])
    axes[2].set_title('Test Ship East Position [m]')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Position (m)')
    axes[2].grid(color='0.8', linestyle='-', linewidth=0.5)
    axes[2].set_xlim(left=0)

    # Plot 4.1: Obsacle east position
    axes[3].plot(os_results_df['time [s]'], os_results_df['east position [m]'])
    axes[3].set_title('Obstacle Ship East Position [m]')
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylabel('Position (m)')
    axes[3].grid(color='0.8', linestyle='-', linewidth=0.5)
    axes[3].set_xlim(left=0)

    # Adjust layout for better spacing
    plt.tight_layout()
    
# Show Plot
plt.show()