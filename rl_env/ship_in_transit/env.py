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
from rl_env.ship_in_transit.reward_functions.evaluation_function import evaluation_function
from rl_env.ship_in_transit.reward_functions import evaluation_item

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

class MultiShipEnv(Env):
    """
    This class is the main class for the Ship-Transit Simulator suited for 
    multi-ship operation. It handles:
    - One ship under test
    - One or more obstacle ships
    """
    def __init__(self, 
                 assets:List[ShipAssets],
                 map: PolygonObstacle,
                 ship_draw:bool,
                 time_since_last_ship_drawing:float,
                 args):
        super().__init__()
        
        # Store args as attribute
        self.args = args
        
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
        self.obs_dim = self.observation_space.shape[0]
        
        # Define action space [route_point_shift] 
        self.action_space = Box(
            low = np.array([-np.pi/6], dtype=np.float32),
            high = np.array([np.pi/6], dtype=np.float32),
        )
        self.action_dim = self.action_space.shape[0]
        
        # Define initial state
        self.initial_states = np.array([self.test.ship_model.north, self.test.ship_model.east, 0.0,
                                       self.obs.ship_model.north, self.obs.ship_model.east, 
                                       self.obs.ship_model.yaw_angle, 0.0, self.obs.ship_model.init_forward_speed], dtype=np.float32)
        self.states = self.initial_states
        
        # Define next state
        self.next_states = self.initial_states
        
        # Store the map class as attribute
        self.map = map 
        
        # Ship drawing configuration
        self.ship_draw = ship_draw
        self.time_since_last_ship_drawing = time_since_last_ship_drawing
    
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
    
    def test_step(self):
        ''' 
            The method is used for stepping up the simulator for the ship under test
        '''          
        # Measure ship position and speed
        north_position = self.test.ship_model.north
        east_position = self.test.ship_model.east
        heading = self.test.ship_model.yaw_angle
        forward_speed = self.test.ship_model.forward_speed
        
        # Find appropriate rudder angle and engine throttle
        rudder_angle = self.test.auto_pilot.rudder_angle_from_sampled_route(
            north_position=north_position,
            east_position=east_position,
            heading=heading
        )
        
        throttle = self.test.throttle_controller.throttle(
            speed_set_point = self.test.desired_forward_speed,
            measured_speed = forward_speed,
            measured_shaft_speed = forward_speed,
        )
            
        ## COLLISION AVOIDANCE
        ####################################################################################################
        collision_risk = evaluation_item.is_collision_imminent(self.states[0:2], self.states[3:5])
        ####################################################################################################
                
        if collision_risk:
            # Reduce throttle
            throttle *= 0.5
            throttle = np.clip(throttle, 0.0, 1.1)

            # Add a small rudder bias to steer away (rudder angle in radians)
            rudder_angle += np.deg2rad(3)                    
            rudder_angle = np.clip(rudder_angle, 
                                   -self.test.auto_pilot.heading_controller.max_rudder_angle, 
                                   self.test.auto_pilot.heading_controller.max_rudder_angle)

            # Optional print for debugging
            # print("Collision Avoidance Activated")
        
        # Update and integrate differential equations for current time step
        self.test.ship_model.store_simulation_data(throttle, 
                                                   rudder_angle,
                                                   self.test.auto_pilot.get_cross_track_error(),
                                                   self.test.auto_pilot.get_heading_error())
        self.test.ship_model.update_differentials(engine_throttle=throttle, rudder_angle=rudder_angle)
        self.test.ship_model.integrate_differentials()
        
        self.test.integrator_term.append(self.test.auto_pilot.navigate.e_ct_int)
        self.test.time_list.append(self.test.ship_model.int.time)
        
        # Get observations
        pos = [self.test.ship_model.north, self.test.ship_model.east, self.test.ship_model.yaw_angle]
        los_ct_error = self.test.ship_model.simulation_results['cross track error [m]'][-1]
        
        # Set the next state, then reset the next_state container to zero
        next_states = [self.ensure_scalar(pos[0]),
                      self.ensure_scalar(pos[1]),
                      self.ensure_scalar(los_ct_error),]
        
        # Step up the simulator
        self.test.ship_model.int.next_time()
        
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
        
        # Compute reward
        pos = [self.obs.ship_model.north, self.obs.ship_model.east, self.obs.ship_model.yaw_angle]
        los_ct_error = self.obs.ship_model.simulation_results['cross track error [m]'][-1]
        
        # Get the next state
        next_states = [self.ensure_scalar(pos[0]),
                      self.ensure_scalar(pos[1]), 
                      self.ensure_scalar(pos[2]),
                      self.ensure_scalar(forward_speed),
                      self.ensure_scalar(los_ct_error),]

        # Step up the simulator
        self.obs.ship_model.int.next_time()
        
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
        termination_cond = evaluation_function(next_states,
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
        
        return next_states, done
    
    def seed(self, seed=None):
        """Set the random seed for reproducibility"""
        self.np_random, seed = seeding.np_random(seed)
        
    # Make sure all values are scalars
    def ensure_scalar(self, x):
        return float(x[0]) if isinstance(x, (np.ndarray, list)) else float(x)