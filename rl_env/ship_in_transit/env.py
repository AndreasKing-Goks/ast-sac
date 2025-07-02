""" 
This modules provide classes for the reinforcement learning environment based on the Ship-Transit Simulator
"""

from gymnasium import Env
from gymnasium.spaces import Box
from gymnasium.utils import seeding

from collections import defaultdict
import numpy as np
import torch

from rl_env.ship_in_transit.sub_systems.ship_model import ShipModel, ShipModelAST
from rl_env.ship_in_transit.sub_systems.controllers import EngineThrottleFromSpeedSetPoint, HeadingByRouteController, HeadingBySampledRouteController
from rl_env.ship_in_transit.sub_systems.obstacle import StaticObstacle, PolygonObstacle
from rl_env.ship_in_transit.evaluation.termination_flags import get_termination_status
from rl_env.ship_in_transit.evaluation import check_condition
from rl_env.ship_in_transit.sub_systems.sbmpc import SBMPC

from rl_env.ship_in_transit.evaluation.reward_function import RewardTracker, get_reward_and_env_info

from dataclasses import dataclass, field
from typing import Union, List

import copy

@dataclass
class ShipAssets:
    ship_model: Union[ShipModel, ShipModelAST]
    throttle_controller: EngineThrottleFromSpeedSetPoint
    auto_pilot: Union[HeadingBySampledRouteController, HeadingByRouteController]
    desired_forward_speed: float
    integrator_term: List[float]
    time_list: List[float]
    type_tag: str
    stop_flag: bool
    init_copy: 'ShipAssets' = field(default=None, repr=False, compare=False)

class MultiShipRLEnv(Env):
    """
    This class is the main class for the Ship-Transit Simulator suited for 
    AST-SAC process with multi-ship operation. It handles:
    - One ship under test
    - One or more obstacle ships
    - Obstacle ships is equipped with intermediate waypoint sampler
    
    To turn on collision avoidance on the ship under test:
    - set collav=None         : No collision avoidance is implemented
    - set collav='simple'     : Simple collision avoidance is implemented
    - set collav='sbmpc'      : SBMPC collision avoidance is implemented
    """
    def __init__(self, 
                 assets:List[ShipAssets],
                 map: PolygonObstacle,
                 ship_draw:bool,
                 collav:str,
                 time_since_last_ship_drawing:float,
                 args,
                 normalize_action=False):
        super().__init__()
        
        # Store args as attribute
        self.args = args
        
        # Set collision avoidance handle
        self.collav = collav
        
        ## Unpack assets [test, obs]
        self.assets = assets
        [self.test, self.obs] = self.assets
        
        # Store initial values for each assets for reset function
        for i, asset in enumerate(self.assets):
            asset.init_copy=copy.deepcopy(asset)
        
        # Define observation space
        # [test_n_pos, test_e_pos, test_los_e_ct \
        #   obs_n_pos, obs_e_pos, obs_heading, obs_los_e_ct, obs_vel]  (8 states)
        self.observation_space = Box(
            low = np.array([0, 0, -3000, 
                            0, 0, -np.pi, -3000, 0], dtype=np.float32),
            high = np.array([10000, 20000, 3000,
                             10000, 20000, np.pi, 3000, 10], dtype=np.float32),
            dtype=np.float32)
        
        self.normalize_action = normalize_action
        
        if self.normalize_action:
            # Normalized scoping angle between -1 and 1
            self.action_space = Box(
                low = np.array([-1.0], dtype=np.float32),
                high = np.array([1.0], dtype=np.float32),
            dtype=np.float32)
        else:
            # Scoping Angle between -30 degree to 30 degree, but in radians
            self.action_space = Box(
                low = np.array([-np.pi/6], dtype=np.float32),
                high = np.array([np.pi/6], dtype=np.float32),
            dtype=np.float32)
        
        # Define initial state
        self.initial_states = np.array([self.test.ship_model.north, self.test.ship_model.east, 0.0,
                                       self.obs.ship_model.north, self.obs.ship_model.east, 
                                       self.obs.ship_model.yaw_angle, 0.0, self.obs.ship_model.forward_speed], dtype=np.float32)
        self.states = self.initial_states
        
        # Define next state
        self.next_states = self.initial_states
        
        # Store the map class as attribute
        self.map = map 
        
        # Ship drawing configuration
        self.ship_draw = ship_draw
        self.time_since_last_ship_drawing = time_since_last_ship_drawing

        # Scenario-Based Model Predictive Controller
        self.sbmpc = SBMPC(tf=1000, dt=20)
        
        ## INTERMEDIATE WAYPOINT SAMPLER PROPERTIES
        # Set the intermediate waypoint sampler parameter
        self.init_get_intermediate_waypoints()
        
        # Set up the waypoint sampling time tracker
        self.waypoint_sampling_times = []
        
        # Set up the Reward Tracker
        self.reward_tracker = RewardTracker()
    
    def init_get_intermediate_waypoints(self):
        # Initiate parameter for intermediate waypoint sampler
        AB_distance_n = self.obs.auto_pilot.navigate.north[-1] - self.obs.auto_pilot.navigate.north[0]
        AB_distance_e = self.obs.auto_pilot.navigate.east[-1] - self.obs.auto_pilot.navigate.east[0]
        
        self.AB_length = np.sqrt(AB_distance_n ** 2 + AB_distance_e ** 2)
        self.AB_segment_length       = self.AB_length / (self.args.sampling_frequency + 1)
        self.AB_north_segment_length = AB_distance_n / (self.args.sampling_frequency + 1)
        self.AB_east_segment_length  = AB_distance_e / (self.args.sampling_frequency + 1) 
        
        AB_alpha = np.arctan2(AB_distance_e, AB_distance_n)
        AB_beta = np.pi/2 - AB_alpha 
        self.omega = np.pi/2 - AB_beta
        
        # Set up recursive variable
        self.y_s = 0
        self.x_s = 0
        self.n_base = self.AB_north_segment_length + self.obs.auto_pilot.navigate.north[0]
        self.e_base = self.AB_east_segment_length + self.obs.auto_pilot.navigate.east[0]
        self.sampling_count = 0
        
        # Set up the ship travel time and travel distance tracker between each waypoints
        self.tracker_active = False # Activate tracking only after taking a init_step, deactivate only after reset()
        self.travel_dist_record= []
        self.travel_time_record= []
        self.travel_dist = 0 # Set to 0 after sampling an intermediate waypoint
        self.travel_time = 0 # Set to 0 after sampling an intermediate waypoint
    
    def normalize_action(self, a_real):
        '''
        Map real action to normalized [-1,1] space
        '''
        return 2.0 * (a_real - self.action_space.low) / (self.action_space.high - self.action_space.low) - 1.0
    
    def denormalize_action(self, a_norm):
        '''
        Map normalized [-1,1] to real action space
        '''
        return (a_norm + 1.0) / 2.0 * (self.action_space.high - self.action_space.low) + self.action_space.low
    
    def get_intermediate_waypoints(self, action):
        '''
        Take action as input (scoping angle).
        Convert the action as the next intermediate waypoints.
        
        **ALWAYS takes the unormalized action**
        '''
        # Unpack the action and get the scoping angle \Chi (It is a 1D vector)
        # scoping_angle =  action 
        scoping_angle = action[0]  # ----> UNIQUE IMPLEMENTATION FOR HAARNOJA ACTION
        
        # Compute n_s and e_s
        # self.omega = (np.pi/2 - self.AB_beta)
        l_s = np.abs(self.AB_segment_length * np.tan(scoping_angle))
        self.e_s = l_s * np.cos(self.omega)
        self.n_s = l_s * np.sin(self.omega)
        
        # Compute x_base and y_base
        # Positive sampling angle turn left
        if scoping_angle > 0:
            self.e_s *= -1
        else:
            self.n_s *= -1
        
        # Compute next route coordinate
        route_coord_n = self.n_base + self.n_s
        route_coord_e = self.e_base + self.e_s
        
        # Update new base for the next route coordinate
        self.n_base = route_coord_n + self.AB_north_segment_length
        self.e_base = route_coord_e + self.AB_east_segment_length
        
        # Repack into simulation input
        intermediate_waypoints = [route_coord_n, route_coord_e]
        
        return intermediate_waypoints
    
    def reset(self,
              action=None):
        ''' 
            Reset all of the ship environment inside the assets container.
            
            Immediately call upon init_step() and init_get_intermediate_waypoint() method
            
            Return the initial state
            
            Note:
            When the action is not None, it means we immediately sample
            an intermediate waypoints for the obstacle ship to use
        '''
        # Reset the assets
        for i, ship in enumerate(self.assets):
            # Call upon the copied initial values
            init = ship.init_copy
            
            #  Reset the ship simulator
            ship.ship_model.reset()
            
            # Reset the ship throttle controller
            ship.throttle_controller.reset() 
            
            # Reset the autopilot controlller
            ship.auto_pilot.reset()
            
            # Reset parameters and lists
            ship.desired_forward_speed = init.desired_forward_speed
            ship.integrator_term = copy.deepcopy(init.integrator_term)
            ship.time_list = copy.deepcopy(init.time_list)
            
            # Reset the stop flag
            ship.stop_flag = False
        
        # Reset the intermediate waypoint converter
        self.init_get_intermediate_waypoints()
        
        # Reset the waypoint sampling time tracker
        self.waypoint_sampling_times = []
        
        # Place the assets in the simulator
        self.init_step(action)
        
        return self.initial_states
    
    def init_step(self,
                  action=None):
        ''' 
            The initial step to place the ships at their initial states
            and to initiate the controller before running the simulator.
            
            Note:
            When the action is not None, it means we immediately sample
            an intermediate waypoints for the obstacle ship to use
        '''
        # For all assets
        for ship in self.assets:
            # Measure ship position and speed
            north_position = ship.ship_model.north
            east_position = ship.ship_model.east
            heading = ship.ship_model.yaw_angle
            forward_speed = ship.ship_model.forward_speed
        
            # Find appropriate rudder angle and engine throttle
            rudder_angle = ship.auto_pilot.rudder_angle_from_sampled_route(
                north_position=north_position,
                east_position=east_position,
                heading=heading
            )
        
            throttle = ship.throttle_controller.throttle(
                speed_set_point = ship.desired_forward_speed,
                measured_speed = forward_speed,
                measured_shaft_speed = forward_speed
            )
            
            # Store simulation data after init step
            ship.ship_model.store_simulation_data(throttle, 
                                                  rudder_angle,
                                                  ship.auto_pilot.get_cross_track_error(),
                                                  ship.auto_pilot.get_heading_error())
            
            # Step
            ship.ship_model.update_differentials(engine_throttle=throttle, rudder_angle=rudder_angle)
            ship.ship_model.integrate_differentials()
            
            # Progress time variable to the next time step
            ship.ship_model.int.next_time()
        
        # # Then directly sample intermediate waypoints and let the obstacle ship used it
        # scoping_angle = action
        
        # # If the action is not none and need the action is already normalized
        # if self.normalize_action and action is not None:
        #     scoping_angle = self.denormalize_action(action)
        
        # # If the action is not none and not normalized anymore
        # if action is not None:
        #     self.obs_ship_uses_scoping_angle(scoping_angle)
                    
        # Activate travel distance and travel time tracker after placing the assets in the simulator
        self.tracker_active = True
        
    
    def test_step(self):
        ''' 
            The method is used for stepping up the simulator for the ship under test
        '''          
        # Measure ship position and speed
        north_position = self.test.ship_model.north
        east_position = self.test.ship_model.east
        heading = self.test.ship_model.yaw_angle
        forward_speed = self.test.ship_model.forward_speed
        
        # Keep it, even when sbmpc is disabled
        speed_factor, desired_heading_offset = 1.0, 0.0

        ## COLLISION AVOIDANCE - SBMPC
        ####################################################################################################
        if self.collav == 'sbmpc':
            # Get desired heading and speed for collav
            self.next_wpt, self.prev_wpt = self.test.auto_pilot.navigate.next_wpt(self.test.auto_pilot.next_wpt, north_position, east_position)
            chi_d = self.test.auto_pilot.navigate.los_guidance(self.test.auto_pilot.next_wpt, north_position, east_position)
            u_d = self.test.desired_forward_speed

            # Get OS state required for SBMPC
            os_state = np.array([self.assets[0].ship_model.east,            # x
                                 self.assets[0].ship_model.north,           # y
                                 -self.assets[0].ship_model.yaw_angle,      # Same reference angle but clockwise positive
                                 self.assets[0].ship_model.forward_speed,   # u
                                 self.assets[0].ship_model.sideways_speed,  # v
                                 self.assets[0].ship_model.yaw_rate         # Unused
                                ])
            
            # Get dynamic obstacles info required for SBMPC
            do_list = [
                (i, np.array([asset.ship_model.east, asset.ship_model.north, -asset.ship_model.yaw_angle, asset.ship_model.forward_speed, asset.ship_model.sideways_speed]), None, asset.ship_model.ship_config.length_of_ship, asset.ship_model.ship_config.width_of_ship) for i, asset in enumerate(self.assets[1::]) # first asset is own ship
            ]

            speed_factor, desired_heading_offset = self.sbmpc.get_optimal_ctrl_offset(
                    u_d=u_d,
                    chi_d=-chi_d,
                    os_state=os_state,
                    do_list=do_list
                )
            
            # Note: self.sbmpc.is_stephen_useful() -> bool can be used to know whether or not the SBMPC colav algorithm is currently active
        #################################################################################################### 

        # Find appropriate rudder angle and engine throttle
        rudder_angle = self.test.auto_pilot.rudder_angle_from_sampled_route(
            north_position=north_position,
            east_position=east_position,
            heading=heading,
            desired_heading_offset=-desired_heading_offset # Negative sign because pos==clockwise in sim
        )

        throttle = self.test.throttle_controller.throttle(
            speed_set_point = self.test.desired_forward_speed * speed_factor,
            measured_speed = forward_speed,
            measured_shaft_speed = forward_speed,
        )

        # COLLISION AVOIDANCE - SIMPLE
        ###################################################################################################
        if self.collav == 'simple':            
            collision_risk = check_condition.is_collision_imminent(self.states[0:2], self.states[3:5])
                
            if collision_risk:
                # Reduce throttle
                throttle *= 0.5
                throttle = np.clip(throttle, 0.0, 1.1)

                # Add a small rudder bias to steer away (rudder angle in radians)
                # rudder_angle += np.deg2rad(15) 
                rudder_angle += np.deg2rad(-15) # This triggers SHIP COLLISION
                rudder_angle = np.clip(rudder_angle, 
                                       -self.test.auto_pilot.heading_controller.max_rudder_angle, 
                                        self.test.auto_pilot.heading_controller.max_rudder_angle)
        ###################################################################################################
        
        # Update and integrate differential equations for current time step
        self.test.ship_model.store_simulation_data(throttle, 
                                                   rudder_angle,
                                                   self.test.auto_pilot.get_cross_track_error(),
                                                   self.test.auto_pilot.get_heading_error())
        self.test.ship_model.update_differentials(engine_throttle=throttle, rudder_angle=rudder_angle)
        self.test.ship_model.integrate_differentials()
        
        self.test.integrator_term.append(self.test.auto_pilot.navigate.e_ct_int)
        self.test.time_list.append(self.test.ship_model.int.time)
        
        # Step up the simulator
        self.test.ship_model.int.next_time()
        
        # Get observations
        pos = [self.test.ship_model.north, self.test.ship_model.east, self.test.ship_model.yaw_angle]
        los_ct_error = self.test.ship_model.simulation_results['cross track error [m]'][-1]
        
        # Set the next state, then reset the next_state container to zero
        next_states = np.array([self.ensure_scalar(pos[0]),
                                self.ensure_scalar(pos[1]),
                                self.ensure_scalar(los_ct_error)],
                               dtype=np.float32)
        
        return next_states
    
    def obs_step(self,
                 scoping_angle=None):
        ''' 
            The method is used for stepping up the simulator for the obstacle ship.
            
            Obstacle ship is equipped with the intermediate waypoint sampler.
            
            Can still be used without intermediate waypoint sample
            
            ***obs_step() ALWAYS takes the unormalized action.***
        '''          
        if self.obs.stop_flag:
            # Ship reached endpoint, keep time aligned but don't integrate
            self.obs.ship_model.store_last_simulation_data()
            self.obs.ship_model.int.next_time()  # still progress time
            
            # Get observations
            pos = [self.obs.ship_model.north, self.obs.ship_model.east, self.obs.ship_model.yaw_angle]
            los_ct_error = self.obs.ship_model.simulation_results['cross track error [m]'][-1]
        
            # Store integrator term and timestamp
            self.obs.integrator_term.append(self.obs.auto_pilot.navigate.e_ct_int)
            self.obs.time_list.append(self.obs.ship_model.int.time)
        
            # Step up the simulator
            self.obs.ship_model.int.next_time()
            
            # Stop the ship
            forward_speed = self.obs.ship_model.forward_speed
            forward_speed = 0
        
            # Set the next state using the last position of the ship
            next_states = np.array([self.ensure_scalar(pos[0]),
                                    self.ensure_scalar(pos[1]), 
                                    self.ensure_scalar(pos[2]),
                                    self.ensure_scalar(forward_speed),
                                    self.ensure_scalar(los_ct_error)],
                                    dtype=np.float32)
        
            return next_states
        
        # When the obstacle ship use scoping angle
        if scoping_angle:
            # self.obs_ship_uses_scoping_angle(scoping_angle)
            
            # Then record the time when the obstacle ship sample the scoping angle
            self.waypoint_sampling_times.append(self.obs.ship_model.int.time)
            
        # Measure ship position and speed
        north_position = self.obs.ship_model.north
        east_position = self.obs.ship_model.east
        heading = self.obs.ship_model.yaw_angle
        forward_speed = self.obs.ship_model.forward_speed
        
        # Find appropriate rudder angle and engine throttle
        rudder_angle = self.obs.auto_pilot.rudder_angle_from_sampled_route(
            north_position=north_position,
            east_position=east_position,
            heading=heading
        )
        
        throttle = self.obs.throttle_controller.throttle(
            speed_set_point = self.obs.desired_forward_speed,
            measured_speed = forward_speed,
            measured_shaft_speed = forward_speed,
        )
        
        # Update and integrate differential equations for current time step
        self.obs.ship_model.store_simulation_data(throttle, 
                                                   rudder_angle,
                                                   self.obs.auto_pilot.get_cross_track_error(),
                                                   self.obs.auto_pilot.get_heading_error())
        self.obs.ship_model.update_differentials(engine_throttle=throttle, rudder_angle=rudder_angle)
        self.obs.ship_model.integrate_differentials()
        
        self.obs.integrator_term.append(self.obs.auto_pilot.navigate.e_ct_int)
        self.obs.time_list.append(self.obs.ship_model.int.time)
        
        # Step up the simulator
        self.obs.ship_model.int.next_time()
        
        # Compute reward
        pos = [self.obs.ship_model.north, self.obs.ship_model.east, self.obs.ship_model.yaw_angle]
        los_ct_error = self.obs.ship_model.simulation_results['cross track error [m]'][-1]
        
        # Get the next state
        next_states = np.array([self.ensure_scalar(pos[0]),
                                self.ensure_scalar(pos[1]), 
                                self.ensure_scalar(pos[2]),
                                self.ensure_scalar(forward_speed),
                                self.ensure_scalar(los_ct_error)],
                                dtype=np.float32)
        
        # Track travel time and distance between intermediate waypoint sampling
        if self.tracker_active:
            # Increment the travel distance
            travel_dist_north = self.obs.ship_model.simulation_results['north position [m]'][-1] - self.obs.ship_model.simulation_results['north position [m]'][-2]
            travel_dist_east = self.obs.ship_model.simulation_results['east position [m]'][-1] - self.obs.ship_model.simulation_results['east position [m]'][-2]
            self.travel_dist += np.sqrt(travel_dist_north**2 + travel_dist_east**2)
            
            # Increment the travel tim
            self.travel_time += self.obs.ship_model.int.dt
        
        return next_states

    def obs_ship_uses_scoping_angle(self, scoping_angle):
        # Update the sampling counter
        self.sampling_count += 1
        
        # Convert action into intermediate waypoints
        intermediate_waypoints = self.get_intermediate_waypoints(scoping_angle)
            
        # Update intermediate waypoint to the LOS guidance controller
        self.obs.auto_pilot.update_route(intermediate_waypoints)

        # Append the recorded travel time and travel distance to the list
        self.travel_dist_record.append(self.travel_dist)
        self.travel_time_record.append(self.travel_time)
            
        # Set the travel time and travel distance tracker upon RoA visit. Start recounting again.
        self.travel_dist = 0
        self.travel_time = 0
    
    def _step(self, 
             action=None):
        ''' The method is used for stepping up the simulator for all of the assets
            
            Default action is unormalized. 
            Denormalized the action if the default action mode is normalized
            
            No further denormalization is needed.
        '''
        # Unpack the action
        scoping_angle = action
        if self.normalize_action and action is not None:
            scoping_angle = self.denormalize_action(action)
        
        # Do test ship step
        test_next_state = self.test_step()
        
        # Do obstacle ship step, get scoping angle as the action
        obs_next_state = self.obs_step(scoping_angle)
        
        # Apply ship drawing (set as optional function) after stepping
        if self.ship_draw:
            if self.time_since_last_ship_drawing > 30:
                self.test.ship_model.ship_snap_shot()
                self.obs.ship_model.ship_snap_shot()
                self.time_since_last_ship_drawing = 0 # The ship draw timer is reset here
            self.time_since_last_ship_drawing += self.test.ship_model.int.dt
            
        # Gather the necessary state for the SAC
        next_states = np.array([
                                test_next_state[0], # test_n_pos
                                test_next_state[1], # test_e_pos
                                test_next_state[2], # test_e_ct
                                obs_next_state[0],  # obs_n_pos
                                obs_next_state[1],  # obs_e_pos
                                obs_next_state[2],  # obs_heading
                                obs_next_state[3],  # obs_speed
                                obs_next_state[4],  # obs_e_ct
                                ], dtype=np.float32)
        
        # Next states
        self.states = next_states 
        
        # Arguments for evaluation function
        env_args = (self.assets, self.map, self.travel_dist, self.AB_segment_length, self.travel_time)
        if scoping_angle:
            intermediate_waypoints = self.get_intermediate_waypoints(scoping_angle)
        else:
            intermediate_waypoints = None
        
        # Get the reward, and the termination flags
        reward, env_info = get_reward_and_env_info(env_args, 
                                                   intermediate_waypoints,
                                                   reward_tracker=self.reward_tracker)
        
        terminal = env_info['terminal']                                             # Safety violations
        done = env_info['test_ship_stop'] and (not env_info['terminal'])            # Ship under test stopped, but unharmed
        obs_ship_stop = env_info['obs_ship_stop'] and (not env_info['terminal'])    # Obstacle ship stopped, but unharmed
        
        if obs_ship_stop:
            self.obs.stop_flag = True
        
        combined_done = terminal or done
        
        return next_states, reward, combined_done, env_info
    
    def step(self, 
             action):
        ''' The method is used for stepping up the simulator in accordance to the AST-SAC framework
            
            Default action is unormalized. 
            Denormalized the action if the default action mode is normalized
            
            No further denormalization is needed.
            
            NOTES:
            - Next state is the state for the simulator
            - Next observation is the state for the AST-SAC
            - We ONLY keep the next observation for the AST-SAC purposes
            - For plotting, we can just access the simulator state list from the assets alone
        '''
        # First assumption are simulator is not terminated and 
        # obstacle ship haven't reached the next radius of acceptance
        is_reach_roa = False
        combined_done = False
        
        # Set up container
        accumulated_reward = 0
        
        # If the action is not none and need the action is already normalized
        if self.normalize_action and action is not None:
            action = self.denormalize_action(action)
        
        # If the action is not none and not normalized anymore
        if action is not None:
            self.obs_ship_uses_scoping_angle(action)
        
        # If reaching RoA, not done, and within simulation time limit do stepping with action
        # If not, just step the simulator
        while not is_reach_roa and not combined_done:
            # Step up the simulator
            next_states, reward, combined_done, env_info = self._step()
            
            # Accumulate the next reward for the AST-SAC
            accumulated_reward += reward
            
            # Re-checking if the obstacle ship reach the radius of acceptance region
            north_position = self.obs.ship_model.north
            east_position = self.obs.ship_model.east
            obs_pos = [north_position, east_position]

            is_reach_roa = check_condition.is_reach_radius_of_acceptance(self.obs, 
                                                                         obs_pos,
                                                                         r_o_a=self.args.radius_of_acceptance)
            
            # Check if the simulator has stopped or not
            if combined_done:
                # Then, the next states become the next observations, the break the current looping
                next_observations = next_states
                break
            
            # If finally reach radius of acceptance, do final step with action included
            if is_reach_roa:
                next_states, reward, combined_done, env_info = self._step(action)
                
                # Then, accumulate the next reward for the AST-SAC
                accumulated_reward += reward
                
                # Then, the next states become the next observations, the break the current looping
                next_observations = next_states
                break
        
        return next_observations, accumulated_reward, combined_done, env_info
    
    def seed(self, seed=None):
        """Set the random seed for reproducibility"""
        self.np_random, seed = seeding.np_random(seed)
        
    # Make sure all values are scalars
    def ensure_scalar(self, x):
        return float(x[0]) if isinstance(x, (np.ndarray, list)) else float(x)

class MultiShipEnv(Env):
    """
    This class is the main class for the Ship-Transit Simulator suited for 
    multi-ship operation. It handles:
    - One ship under test
    - One or more obstacle ships
    
    To turn on collision avoidance on the ship under test:
    - set collav=None         : No collision avoidance is implemented
    - set collav='simple'     : Simple collision avoidance is implemented
    - set collav='sbmpc'      : SBMPC collision avoidance is implemented
    """
    def __init__(self, 
                 assets:List[ShipAssets],
                 map: PolygonObstacle,
                 ship_draw:bool,
                 collav:str,
                 time_since_last_ship_drawing:float,
                 args):
        super().__init__()
        
        # Store args as attribute
        self.args = args
        
        # Set collision avoidance handle
        self.collav = collav
        
        ## Unpack assets [test, obs]
        self.assets = assets
        [self.test, self.obs] = self.assets
        
        # Store initial values for each assets for reset function
        for i, asset in enumerate(self.assets):
            asset.init_copy=copy.deepcopy(asset)
        
        # Define observation space
        # [test_n_pos, test_e_pos, test_los_e_ct \
        #   obs_n_pos, obs_e_pos, obs_heading, obs_los_e_ct, obs_vel]  (8 states)
        self.observation_space = Box(
            low = np.array([0, 0, -3000, 
                            0, 0, -np.pi, -3000, 0], dtype=np.float32),
            high = np.array([10000, 20000, 3000,
                             10000, 20000, np.pi, 3000, 10], dtype=np.float32),
        )
        self.obsv_dim = self.observation_space.shape[0]
        
        # Define action space [route_point_shift] 
        self.action_space = Box(
            low = np.array([-np.pi/6], dtype=np.float32),
            high = np.array([np.pi/6], dtype=np.float32),
        )
        self.action_dim = self.action_space.shape[0]
        
        # Define initial state
        self.initial_states = np.array([self.test.ship_model.north, self.test.ship_model.east, 0.0,
                                       self.obs.ship_model.north, self.obs.ship_model.east, 
                                       self.obs.ship_model.yaw_angle, 0.0, self.obs.ship_model.forward_speed], dtype=np.float32)
        self.states = self.initial_states
        
        # Define next state
        self.next_states = self.initial_states
        
        # Store the map class as attribute
        self.map = map 
        
        # Ship drawing configuration
        self.ship_draw = ship_draw
        self.time_since_last_ship_drawing = time_since_last_ship_drawing

        # Scenario-Based Model Predictive Controller
        self.sbmpc = SBMPC(tf=1000, dt=20)
    
    def reset(self):
        ''' 
            Reset all of the ship environment inside the assets container.
            
            Immediately call upon init_step() method
            
            Return the initial state
        '''
        # Reset the assets
        for i, ship in enumerate(self.assets):
            # Call upon the copied initial values
            init = ship.init_copy
            
            #  Reset the ship simulator
            ship.ship_model.reset()
            
            # Reset the ship throttle controller
            ship.throttle_controller.reset() 
            
            # Reset the autopilot controlller
            ship.auto_pilot.reset()
            
            # Reset parameters and lists
            ship.desired_forward_speed = init.desired_forward_speed
            ship.integrator_term = copy.deepcopy(init.integrator_term)
            ship.time_list = copy.deepcopy(init.time_list)
            
            # Reset the stop flag
            ship.stop_flag = False
        
        # Place the assets in the simulator
        self.init_step()
        
        return self.initial_states
    
    def init_step(self):
        ''' 
            The initial step to place the ships at their initial states
            and to initiate the controller before running the simulator.
        '''
        # For all assets
        for ship in self.assets:
            # Measure ship position and speed
            north_position = ship.ship_model.north
            east_position = ship.ship_model.east
            heading = ship.ship_model.yaw_angle
            forward_speed = ship.ship_model.forward_speed
        
            # Find appropriate rudder angle and engine throttle
            rudder_angle = ship.auto_pilot.rudder_angle_from_sampled_route(
                north_position=north_position,
                east_position=east_position,
                heading=heading
            )
        
            throttle = ship.throttle_controller.throttle(
                speed_set_point = ship.desired_forward_speed,
                measured_speed = forward_speed,
                measured_shaft_speed = forward_speed
            )
            
            # Step
            ship.ship_model.update_differentials(engine_throttle=throttle, rudder_angle=rudder_angle)
            ship.ship_model.integrate_differentials()
            
            # Progress time variable to the next time step
            ship.ship_model.int.next_time()
    
    def test_step(self):
        ''' 
            The method is used for stepping up the simulator for the ship under test
        '''          
        # Measure ship position and speed
        north_position = self.test.ship_model.north
        east_position = self.test.ship_model.east
        heading = self.test.ship_model.yaw_angle
        forward_speed = self.test.ship_model.forward_speed
        
        # Keep it, even when sbmpc is disabled
        speed_factor, desired_heading_offset = 1.0, 0.0

        ## COLLISION AVOIDANCE - SBMPC
        ####################################################################################################
        if self.collav == 'sbmpc':
            # Get desired heading and speed for collav
            self.next_wpt, self.prev_wpt = self.test.auto_pilot.navigate.next_wpt(self.test.auto_pilot.next_wpt, north_position, east_position)
            chi_d = self.test.auto_pilot.navigate.los_guidance(self.test.auto_pilot.next_wpt, north_position, east_position)
            u_d = self.test.desired_forward_speed

            # Get OS state required for SBMPC
            os_state = np.array([self.assets[0].ship_model.east,            # x
                                 self.assets[0].ship_model.north,           # y
                                 -self.assets[0].ship_model.yaw_angle,      # Same reference angle but clockwise positive
                                 self.assets[0].ship_model.forward_speed,   # u
                                 self.assets[0].ship_model.sideways_speed,  # v
                                 self.assets[0].ship_model.yaw_rate         # Unused
                                ])
            
            # Get dynamic obstacles info required for SBMPC
            do_list = [
                (i, np.array([asset.ship_model.east, asset.ship_model.north, -asset.ship_model.yaw_angle, asset.ship_model.forward_speed, asset.ship_model.sideways_speed]), None, asset.ship_model.ship_config.length_of_ship, asset.ship_model.ship_config.width_of_ship) for i, asset in enumerate(self.assets[1::]) # first asset is own ship
            ]

            speed_factor, desired_heading_offset = self.sbmpc.get_optimal_ctrl_offset(
                    u_d=u_d,
                    chi_d=-chi_d,
                    os_state=os_state,
                    do_list=do_list
                )
            
            # Note: self.sbmpc.is_stephen_useful() -> bool can be used to know whether or not the SBMPC colav algorithm is currently active
        #################################################################################################### 

        # Find appropriate rudder angle and engine throttle
        rudder_angle = self.test.auto_pilot.rudder_angle_from_sampled_route(
            north_position=north_position,
            east_position=east_position,
            heading=heading,
            desired_heading_offset=-desired_heading_offset # Negative sign because pos==clockwise in sim
        )

        throttle = self.test.throttle_controller.throttle(
            speed_set_point = self.test.desired_forward_speed * speed_factor,
            measured_speed = forward_speed,
            measured_shaft_speed = forward_speed,
        )

        # COLLISION AVOIDANCE - SIMPLE
        ###################################################################################################
        if self.collav == 'simple':            
            collision_risk = check_condition.is_collision_imminent(self.states[0:2], self.states[3:5])
                
            if collision_risk:
                # Reduce throttle
                throttle *= 0.5
                throttle = np.clip(throttle, 0.0, 1.1)

                # Add a small rudder bias to steer away (rudder angle in radians)
                # rudder_angle += np.deg2rad(15) 
                rudder_angle += np.deg2rad(-15) # This triggers SHIP COLLISION
                rudder_angle = np.clip(rudder_angle, 
                                       -self.test.auto_pilot.heading_controller.max_rudder_angle, 
                                        self.test.auto_pilot.heading_controller.max_rudder_angle)
        ###################################################################################################
        
        # Update and integrate differential equations for current time step
        self.test.ship_model.store_simulation_data(throttle, 
                                                   rudder_angle,
                                                   self.test.auto_pilot.get_cross_track_error(),
                                                   self.test.auto_pilot.get_heading_error())
        self.test.ship_model.update_differentials(engine_throttle=throttle, rudder_angle=rudder_angle)
        self.test.ship_model.integrate_differentials()
        
        self.test.integrator_term.append(self.test.auto_pilot.navigate.e_ct_int)
        self.test.time_list.append(self.test.ship_model.int.time)
        
        # Step up the simulator
        self.test.ship_model.int.next_time()
        
        # Get observations
        pos = [self.test.ship_model.north, self.test.ship_model.east, self.test.ship_model.yaw_angle]
        los_ct_error = self.test.ship_model.simulation_results['cross track error [m]'][-1]
        
        # Set the next state, then reset the next_state container to zero
        next_states = [self.ensure_scalar(pos[0]),
                      self.ensure_scalar(pos[1]),
                      self.ensure_scalar(los_ct_error),]
        
        return next_states
    
    def obs_step(self):
        ''' 
            The method is used for stepping up the simulator for the obstacle ship.
        '''          
        if self.obs.stop_flag:
            # Ship reached endpoint, keep time aligned but don't integrate
            self.obs.ship_model.store_last_simulation_data()
            self.obs.ship_model.int.next_time()  # still progress time
            
            # Get observations
            pos = [self.obs.ship_model.north, self.obs.ship_model.east, self.obs.ship_model.yaw_angle]
            los_ct_error = self.obs.ship_model.simulation_results['cross track error [m]'][-1]
        
            # Store integrator term and timestamp
            self.obs.integrator_term.append(self.obs.auto_pilot.navigate.e_ct_int)
            self.obs.time_list.append(self.obs.ship_model.int.time)
        
            # Step up the simulator
            self.obs.ship_model.int.next_time()
            
            # Stop the ship
            forward_speed = self.obs.ship_model.forward_speed
            forward_speed = 0
        
            # Set the next state using the last position of the ship
            next_states = [self.ensure_scalar(pos[0]),
                          self.ensure_scalar(pos[1]), 
                          self.ensure_scalar(pos[2]),
                          self.ensure_scalar(forward_speed),
                          self.ensure_scalar(los_ct_error),]
        
            return next_states
        
        # Measure ship position and speed
        north_position = self.obs.ship_model.north
        east_position = self.obs.ship_model.east
        heading = self.obs.ship_model.yaw_angle
        forward_speed = self.obs.ship_model.forward_speed
        
        # Find appropriate rudder angle and engine throttle
        rudder_angle = self.obs.auto_pilot.rudder_angle_from_sampled_route(
            north_position=north_position,
            east_position=east_position,
            heading=heading
        )
        
        throttle = self.obs.throttle_controller.throttle(
            speed_set_point = self.obs.desired_forward_speed,
            measured_speed = forward_speed,
            measured_shaft_speed = forward_speed,
        )
        
        # Update and integrate differential equations for current time step
        self.obs.ship_model.store_simulation_data(throttle, 
                                                   rudder_angle,
                                                   self.obs.auto_pilot.get_cross_track_error(),
                                                   self.obs.auto_pilot.get_heading_error())
        self.obs.ship_model.update_differentials(engine_throttle=throttle, rudder_angle=rudder_angle)
        self.obs.ship_model.integrate_differentials()
        
        self.obs.integrator_term.append(self.obs.auto_pilot.navigate.e_ct_int)
        self.obs.time_list.append(self.obs.ship_model.int.time)
        
        # Step up the simulator
        self.obs.ship_model.int.next_time()
        
        # Compute reward
        pos = [self.obs.ship_model.north, self.obs.ship_model.east, self.obs.ship_model.yaw_angle]
        los_ct_error = self.obs.ship_model.simulation_results['cross track error [m]'][-1]
        
        # Get the next state
        next_states = [self.ensure_scalar(pos[0]),
                      self.ensure_scalar(pos[1]), 
                      self.ensure_scalar(pos[2]),
                      self.ensure_scalar(forward_speed),
                      self.ensure_scalar(los_ct_error),]
        
        return next_states
    
    def step(self):
        ''' The method is used for stepping up the simulator for all the reinforcement
            learning assets
        '''
        # Do test ship step
        test_next_state = self.test_step()
        
        # Do obstacle ship step
        obs_next_state = self.obs_step()
        
        # Apply ship drawing (set as optional function) after stepping
        if self.ship_draw:
            if self.time_since_last_ship_drawing > 30:
                self.test.ship_model.ship_snap_shot()
                self.obs.ship_model.ship_snap_shot()
                self.time_since_last_ship_drawing = 0 # The ship draw timer is reset here
            self.time_since_last_ship_drawing += self.test.ship_model.int.dt
            
        # Gather the necessary state for the SAC
        next_states = [
                      test_next_state[0], # test_n_pos
                      test_next_state[1], # test_e_pos
                      test_next_state[2], # test_e_ct
                      obs_next_state[0],  # obs_n_pos
                      obs_next_state[1],  # obs_e_pos
                      obs_next_state[2],  # obs_heading
                      obs_next_state[3],  # obs_speed
                      obs_next_state[4],  # obs_e_ct
                      ]
        
        # Next states
        self.states = next_states 
        
        # Arguments for evaluation function
        test_route_end   = [self.test.auto_pilot.navigate.north[-1], self.test.auto_pilot.navigate.east[-1]]
        obs_route_end    = [self.obs.auto_pilot.navigate.north[-1], self.obs.auto_pilot.navigate.east[-1]]
        map_obj          = self.map
        test_ship_length = self.test.ship_model.l_ship
        obs_ship_length  = self.obs.ship_model.l_ship
        
        # Evaluate the current simulator states
        termination_cond = get_termination_status(next_states,
                                                  test_route_end,
                                                  obs_route_end,
                                                  map_obj,
                                                  test_ship_length,
                                                  obs_ship_length)
        
        # Unpack the termination condition
        test_termination_cond = termination_cond[0:4]
        join_termination_cond = termination_cond[4:6]
        obs_termination_cond  = termination_cond[6:10]
        
        test_is_reached, test_is_outside, test_is_grounded, test_is_failed_nav = test_termination_cond
        is_near_collision, is_collision = join_termination_cond
        obs_is_reached, obs_is_outside, obs_is_grounded, obs_is_failed_nav = obs_termination_cond
        
        # Set the termination condition
        done = test_is_reached or test_is_outside or test_is_grounded or test_is_failed_nav or is_collision \
            or obs_is_grounded or obs_is_failed_nav
        
        # Set the flag to stop integrating obstacle ship dynamics when
        # - obstacle ship reaches its destination
        # - obstacle ship exits the map
        stop_int_obs = obs_is_outside and obs_is_reached
        
        if stop_int_obs:
            self.obs.stop_flag = True 
        
        return next_states, done, termination_cond
    
    def seed(self, seed=None):
        """Set the random seed for reproducibility"""
        self.np_random, seed = seeding.np_random(seed)
        
    # Make sure all values are scalars
    def ensure_scalar(self, x):
        return float(x[0]) if isinstance(x, (np.ndarray, list)) else float(x)