### IMPORT SIMULATOR ENVIRONMENTS
from run_colav.env import MultiShipEnv, ShipAssets

from run_colav.ship_in_transit.sub_systems.ship_model import  ShipConfiguration, EnvironmentConfiguration, SimulationConfiguration, SimpleShipModel
from run_colav.ship_in_transit.sub_systems.ship_engine import RudderConfiguration
from run_colav.ship_in_transit.sub_systems.LOS_guidance import LosParameters
from run_colav.ship_in_transit.sub_systems.obstacle import StaticObstacle, PolygonObstacle
from run_colav.ship_in_transit.sub_systems.controllers import SpeedControllerGains, HeadingControllerGains, ThrustFromSpeedSetPoint, HeadingBySampledRouteController
from run_colav.ship_in_transit.utils.print_termination import print_termination

## IMPORT FUNCTIONS
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
parser.add_argument('--time_step', type=int, default=30, metavar='TIMESTEP',
                    help='ENV: time step size in second for ship transit simulator (default: 2)')
parser.add_argument('--radius_of_acceptance', type=int, default=300, metavar='ROA',
                    help='ENV: radius of acceptance for LOS algorithm (default: 300)')
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

args = parser.parse_args()

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

rudder_config = RudderConfiguration(
                                    rudder_angle_to_sway_force_coefficient=50e3,
                                    rudder_angle_to_yaw_force_coefficient=500e3,
                                    max_rudder_angle_degrees=30
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
test_ship = SimpleShipModel(ship_config=ship_config,
                            rudder_config=rudder_config,
                            environment_config=env_config,
                            simulation_config=ship_in_test_simu_setup)

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
obs_ship = SimpleShipModel(ship_config=ship_config,
                           rudder_config=rudder_config,
                           environment_config=env_config,
                           simulation_config=ship_as_obstacle_simu_setup)

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
max_test_ship_thrust = np.inf
test_ship_thrust_controller_gains = SpeedControllerGains(
    kp=150, ki=150, kd=75
)
test_ship_thrust_controller = ThrustFromSpeedSetPoint(
    gains=test_ship_thrust_controller_gains,
    max_thrust=max_test_ship_thrust,
    time_step=args.time_step
)

test_route_filename = 'own_ship_route.txt'
test_route_name = get_data_path(test_route_filename)
test_heading_controller_gains = HeadingControllerGains(kp=.5, ki=0.01, kd=84)
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
    max_rudder_angle=np.deg2rad(rudder_config.max_rudder_angle_degrees),
    num_of_samplings=2
)
test_desired_forward_speed =4.5

test_integrator_term = []
test_times = []

## Set the throttle and autopilot controllers for the obstacle ship
max_obs_ship_thrust = np.inf
obs_ship_thrust_controller_gains = SpeedControllerGains(
    kp=.025, ki=700.5, kd=550.5
)
obs_ship_thrust_controller = ThrustFromSpeedSetPoint(
    gains=obs_ship_thrust_controller_gains,
    max_thrust=max_obs_ship_thrust,
    time_step=args.time_step
)

obs_route_filename = 'obs_ship_route.txt'
obs_route_name = get_data_path(obs_route_filename)
obs_heading_controller_gains = HeadingControllerGains(kp=.65, ki=0.001, kd=50)
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
    max_rudder_angle=np.deg2rad(rudder_config.max_rudder_angle_degrees),
    num_of_samplings=2
)
obs_desired_forward_speed = 4.0 # 5.5 immediate near collision.

obs_integrator_term = []
obs_times = []

# Wraps simulation objects based on the ship type using a dictionary
test = ShipAssets(
    ship_model=test_ship,
    speed_controller=test_ship_thrust_controller,
    auto_pilot=test_auto_pilot,
    desired_forward_speed=test_desired_forward_speed,
    integrator_term=test_integrator_term,
    time_list=test_times,
    stop_flag=False,
    type_tag='test_ship'
)

obs = ShipAssets(
    ship_model=obs_ship,
    speed_controller=obs_ship_thrust_controller,
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
env = MultiShipEnv(assets=assets,
                   map=map,
                   args=args)


print('-------------------------------------------------')

# Test the simulation step up using policy's action sampling or direct action manipulation
test1 = True
# test1 = False

if test1:
 
    print('Test and plot step up behaviour')
    print('sampling count before reset:', env.sampling_count)
    init_observations = env.reset()                                                    # First reset
    

    action = env.do_normalize_action(np.deg2rad(-15))
    print('First action', np.rad2deg(env.do_denormalize_action(action)))
    next_observations, combined_done, env_info = env.step(action)  # Step up
    print('sampling count after first step:', env.sampling_count)  
    

    action = env.do_normalize_action(np.deg2rad(15))
    print('Second action', np.rad2deg(env.do_denormalize_action(action)))
    next_observations, combined_done, env_info = env.step(action)  # Step up
    print('sampling count after second step:', env.sampling_count)
    

    action = env.do_normalize_action(np.deg2rad(15))
    print('Third action', np.rad2deg(env.do_denormalize_action(action)))
    next_observations, combined_done, env_info = env.step(action)  # Step up
    print('sampling count after third step:', env.sampling_count)
    

    action = env.do_normalize_action(np.deg2rad(-15))
    print('Fourth action', np.rad2deg(env.do_denormalize_action(action)))
    next_observations, combined_done, env_info = env.step(action)  # Step up
    print('sampling count after fourth step:', env.sampling_count)
    
         
    action = env.do_normalize_action(np.deg2rad(-15))
    print('Fourth action', np.rad2deg(env.do_denormalize_action(action)))
    next_observations, combined_done, env_info = env.step(action)  # Step up
    print('sampling count after fifth step:', env.sampling_count)
        
         
    action = env.do_normalize_action(np.deg2rad(15))
    print('Fourth action', np.rad2deg(env.do_denormalize_action(action)))
    next_observations, combined_done, env_info = env.step(action)  # Step up
    print('sampling count after sixth step:', env.sampling_count)


    action = env.do_normalize_action(np.deg2rad(15))
    print('Fourth action', np.rad2deg(env.do_denormalize_action(action)))
    next_observations, combined_done, env_info = env.step(action)  # Step up
    print('sampling count after seventh step:', env.sampling_count)

        
    action = env.do_normalize_action(np.deg2rad(-15))
    print('Fourth action', np.rad2deg(env.do_denormalize_action(action)))
    next_observations, combined_done, env_info = env.step(action)  # Step up
    print('sampling count after eighth step:', env.sampling_count)

        
    action = env.do_normalize_action(np.deg2rad(-15))
    print('Fourth action', np.rad2deg(env.do_denormalize_action(action)))
    next_observations, combined_done, env_info = env.step(action)  # Step up
    print('sampling count after ninth step:', env.sampling_count)

    # Get the simulation results for all assets
    ts_results_df = pd.DataFrame().from_dict(env.test.ship_model.simulation_results)
    os_results_df = pd.DataFrame().from_dict(env.obs.ship_model.simulation_results)

    # Get the time when the sampling begin, failure modes and the waypoints
    waypoint_sampling_times = env.waypoint_sampling_times
    
    # Print
    print('')
    print('---')
    print('Waypoint sampling time record:', waypoint_sampling_times)
    print('---')
    print('after step route north:', env.obs.auto_pilot.navigate.north)
    print('after step route east :', env.obs.auto_pilot.navigate.east)
    print('---')
    print(env_info['events'])

    # Plot 1: Overall process plot
    plot_1 = False
    # plot_1 = True

    # Plot 2: Status plot
    plot_2 = False
    # plot_2 = True
    
    # For animation
    animation = False
    animation = True

    if animation:
        test_route = {'east': test.auto_pilot.navigate.east, 'north': test.auto_pilot.navigate.north}
        obs_route = {'east': obs.auto_pilot.navigate.east, 'north': obs.auto_pilot.navigate.north}
        waypoint_sampling_times = env.waypoint_sampling_times
        waypoints = list(zip(obs.auto_pilot.navigate.east, obs.auto_pilot.navigate.north))
        map_obj = map
        radius_of_acceptance = args.radius_of_acceptance
        timestamps = test.time_list
        interval = args.time_step * 1000
        test_drawer=test.ship_model.draw
        obs_drawer=obs.ship_model.draw
        test_headings=ts_results_df["yaw angle [deg]"]
        obs_headings=os_results_df["yaw angle [deg]"]
        is_collision_imminent_list = env.is_collision_imminent_list
        is_collision_list = env.is_collision_list
        
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
        ani_dir = os.path.join('animation', 'FOR_STUDENT')
        filename  = "trajectory.mp4"
        video_path = os.path.join(ani_dir, filename)
        fps = 480
        
        # Create the output directory if it doesn't exist
        os.makedirs(ani_dir, exist_ok=True)
        
        # animator.save(video_path, fps)
        animator.run(fps)

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