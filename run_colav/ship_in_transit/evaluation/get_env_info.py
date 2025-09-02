'''
Reward Functions for AST-SAC
'''

import numpy as np

from run_colav.ship_in_transit.evaluation.check_condition import (is_reaches_endpoint,
                                                               is_pos_outside_horizon,
                                                               is_pos_inside_obstacles,
                                                               is_ship_navigation_failure,
                                                               is_route_inside_obstacles,
                                                               is_route_outside_horizon,
                                                               is_collision_imminent,
                                                               is_ship_collision,
                                                               is_sample_travel_dist_too_far,
                                                               is_sample_travel_time_too_long,
                                                               is_within_simu_time_limit)
from run_colav.ship_in_transit.utils.compute_distance import (get_distance,
                                                           get_distance_and_true_encounter_type, 
                                                           get_distance_and_encounter_type)
from dataclasses import dataclass, field

@dataclass
class RewardTracker:
    ship_collision              : list = field(default_factory=list)
    test_ship_grounding         : list = field(default_factory=list)
    test_ship_nav_failure       : list = field(default_factory=list)
    obs_ship_grounding          : list = field(default_factory=list)
    obs_ship_nav_failure        : list = field(default_factory=list)
    from_test_ship              : list = field(default_factory=list)
    from_obs_ship               : list = field(default_factory=list)
    total                       : list = field(default_factory=list)
    
    def update(self, 
               r_ship_collision,
               r_test_ship_grounding,
               r_test_ship_nav_failure,
               r_obs_ship_grounding,
               r_obs_ship_nav_failure,
               r_total):
        self.ship_collision.append(r_ship_collision)
        self.test_ship_grounding.append(r_test_ship_grounding)
        self.test_ship_nav_failure.append(r_test_ship_nav_failure)
        self.obs_ship_grounding.append(r_obs_ship_grounding)
        self.obs_ship_nav_failure.append(r_obs_ship_nav_failure)
        self.from_test_ship.append(r_test_ship_grounding + r_test_ship_nav_failure)
        self.from_obs_ship.append(r_obs_ship_grounding + r_obs_ship_nav_failure)
        self.total.append(r_total)
        
    def update_r_total_only(self, r_total):
        self.total.append(r_total)
        
def get_env_info(env_args,  
                 use_relative_bearing = True):
    '''
    env_args contains:
    - assets: To get the simulation states for both ship under test and obstacle ship(s)
    - map_obj: To get the positional information based on the map object used for the simulator
    - travel_dist, traj_segment_length, travel_time:
      To get the information about obstacle ship navigational failure potential
    '''
    ## Unpack env_args
    assets, map_obj, travel_dist, traj_segment_length, travel_time = env_args
    
    ## Unpack the ship assets
    test, obs = assets
    
    ## Get the ship under test and the obstacle ship position and cross track error
    test_n_pos       = test.ship_model.north
    test_e_pos       = test.ship_model.east
    test_pos         = [test_n_pos, test_e_pos]
    test_heading     = test.ship_model.yaw_angle
    test_e_ct        = test.ship_model.simulation_results['cross track error [m]'][-1]
    test_ship_length = test.ship_model.l_ship
    
    obs_n_pos       = obs.ship_model.north
    obs_e_pos       = obs.ship_model.east
    obs_pos         = [obs_n_pos, obs_e_pos]
    obs_heading     = obs.ship_model.yaw_angle
    obs_e_ct        = obs.ship_model.simulation_results['cross track error [m]'][-1]
    obs_ship_length = obs.ship_model.l_ship
    
    ## COMPUTE ARGUMENTS FOR THE REWARD FUNCTIONS
    # Ship under test to obstacle design and collision flag
    if use_relative_bearing:
        test_to_obs_distance, encounter_type = get_distance_and_encounter_type(test_pos, 
                                                                               test_heading, 
                                                                               obs_pos, 
                                                                               obs_heading)
    else:
        test_to_obs_distance, encounter_type = get_distance_and_true_encounter_type(test_pos, 
                                                                                    test_heading, 
                                                                                    obs_pos,
                                                                                    obs_heading)
    is_collision =  is_ship_collision(test_pos, obs_pos)    
    
    # Ship under test to ground distance and grounding flag
    test_to_ground_distance = map_obj.obstacles_distance(test_pos[0], test_pos[1])
    is_test_grounding = is_pos_inside_obstacles(map_obj, test_pos, test_ship_length)
    
    # Obstacle ship to ground distance and grounding flag
    obs_to_ground_distance = map_obj.obstacles_distance(obs_pos[0], obs_pos[1])
    is_obs_grounding = is_pos_inside_obstacles(map_obj, obs_pos, obs_ship_length)
    
    # Get navigation failure flags for both ship under test and obstacle ship
    test_e_ct_threshold = 3000
    obs_e_ct_threshold  = 500
    is_test_nav_failure = is_ship_navigation_failure(test_e_ct, e_tol=test_e_ct_threshold)
    is_obs_nav_failure = any([is_sample_travel_dist_too_far(travel_dist, traj_segment_length),
                              is_sample_travel_time_too_long(travel_time),
                              is_ship_navigation_failure(obs_e_ct, e_tol=obs_e_ct_threshold)])
    
    # Compute ships collision reward. Get the termination status
    termination_1 = is_collision
    
    # Compute test ship grounding reward. Get the termination status
    termination_2 = is_test_grounding
    
    # Compute test ship navigation failure reward. get the termination status
    termination_3 = is_test_nav_failure
    
    # Compute obstacle ship grounding reward. Get the termination status
    termination_4 = is_obs_grounding
    
    # Compute obstacle ship navigation failure reward. Get the termination status
    termination_5 = is_obs_nav_failure
    
    ## GET THE TERMINATION AND STOP OBSTACLE FLAG FOR NON-REWARD EVALUATION FUNCTION
    # Compute the necessary argument
    test_route_end   = [test.auto_pilot.navigate.north[-1], test.auto_pilot.navigate.east[-1]]
    obs_route_end    = [obs.auto_pilot.navigate.north[-1], obs.auto_pilot.navigate.east[-1]]
    
    ## Get the termination status for a non meaningful end
    # Ship under test reaches endpoint or goes outside the map
    termination_6 = is_reaches_endpoint(test_route_end, test_pos)
    termination_7 = is_pos_outside_horizon(map_obj, test_pos, test_ship_length)
    
    # Obstacle ship reaches endpoint or goes outside the map
    termination_8 = is_reaches_endpoint(obs_route_end, obs_pos)
    termination_9 = is_pos_outside_horizon(map_obj, obs_pos, obs_ship_length)
    termination_10 = is_within_simu_time_limit(test) # Following ship under test
    
    # Get env_info
    env_info = {
        'events'            : '',
        'terminal'          : False,
        'test_ship_stop'    : False,
        'obs_ship_stop'     : False,
    }
    
    # Initialize log containers
    terminal_flags = []
    test_ship_stop_flags = []
    obs_ship_stop_flags = []

    # Initialize event list
    events = ''

    if termination_1:
        events += 'Ships collision!'
        terminal_flags.append(True)
        test_ship_stop_flags.append(True)
        obs_ship_stop_flags.append(True)

    if termination_2:
        events += '|Ship under test experiences grounding!|'
        terminal_flags.append(True)
        test_ship_stop_flags.append(True)
        obs_ship_stop_flags.append(False)

    if termination_3:
        events += '|Ship under test suffers navigational failure!|'
        terminal_flags.append(True)
        test_ship_stop_flags.append(True)
        obs_ship_stop_flags.append(False)

    if termination_4:
        events += '|Obstacle ship experiences grounding!|'
        terminal_flags.append(True)
        test_ship_stop_flags.append(False)
        obs_ship_stop_flags.append(True)

    if termination_5:
        events += '|Obstacle ship suffers navigational failure!|'
        terminal_flags.append(True)
        test_ship_stop_flags.append(False)
        obs_ship_stop_flags.append(True)

    if termination_6:
        events += '|Ship under test reaches its final destination!|'
        terminal_flags.append(False)
        test_ship_stop_flags.append(True)
        obs_ship_stop_flags.append(False)

    if termination_7:
        events += '|Ship under test goes outside the map horizon!|'
        terminal_flags.append(False)
        test_ship_stop_flags.append(True)
        obs_ship_stop_flags.append(False)

    if termination_8:
        events += '|Obstacle ship reaches its final destination!|'
        terminal_flags.append(False)
        test_ship_stop_flags.append(False)
        obs_ship_stop_flags.append(True)

    if termination_9:
        events += '|Obstacle ship goes outside the map horizon!|'
        terminal_flags.append(False)
        test_ship_stop_flags.append(False)
        obs_ship_stop_flags.append(True)

    if termination_10:
        events += '|Simulation reaches its time limit|'
        terminal_flags.append(False)
        test_ship_stop_flags.append(True)
        obs_ship_stop_flags.append(True)

    # Final assignment to env_info
    env_info['events'] = events
    env_info['terminal'] = np.any(terminal_flags)
    env_info['test_ship_stop'] = np.any(test_ship_stop_flags)
    env_info['obs_ship_stop'] = np.any(obs_ship_stop_flags)
    
    return env_info