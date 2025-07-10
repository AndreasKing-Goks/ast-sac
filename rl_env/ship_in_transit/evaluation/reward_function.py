'''
Reward Functions for AST-SAC
'''

import numpy as np

from rl_env.reward_designs import (RewardDesign1, 
                                   RewardDesign2, 
                                   RewardDesign3, 
                                   RewardDesign4, 
                                   RewardDesign5, 
                                   RewardDesign6)
from rl_env.ship_in_transit.evaluation.check_condition import (is_reaches_endpoint,
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
from rl_env.ship_in_transit.utils.compute_distance import (get_distance,
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
        
def get_reward_and_env_info(env_args,  
                            reward_tracker:RewardTracker,
                            normalize_reward=False, 
                            use_relative_bearing = True):
    '''
    env_args contains:
    - assets: To get the simulation states for both ship under test and obstacle ship(s)
    - map_obj: To get the positional information based on the map object used for the simulator
    - travel_dist, traj_segment_length, travel_time:
      To get the information about obstacle ship navigational failure potential
    '''
    ## Unpack env_args
    assets, map_obj, travel_dist, traj_segment_length, travel_time, current_step_accumulated_rewards = env_args
    
    ## Unpack the ship assets
    test, obs = assets
    
    # Initiate Reward Tracker
    reward_log = reward_tracker
    
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
    r_ship_collision, termination_1 = ships_collision_reward(test_to_obs_distance, current_step_accumulated_rewards, encounter_type, is_collision)
    
    # Compute test ship grounding reward. Get the termination status
    r_test_ship_grounding, termination_2 = test_ship_grounding_reward(test_to_ground_distance, current_step_accumulated_rewards, is_test_grounding)
    
    # Compute test ship navigation failure reward. get the termination status
    r_test_ship_nav_failure, termination_3 = test_ship_nav_failure_reward(test_e_ct,
                                                                          current_step_accumulated_rewards, 
                                                                          is_test_nav_failure,
                                                                          e_ct_threshold=test_e_ct_threshold)
    
    # Compute obstacle ship grounding reward. Get the termination status
    r_obs_ship_grounding, termination_4 = obs_ship_grounding_reward(obs_to_ground_distance, current_step_accumulated_rewards, is_obs_grounding)
    
    # Compute obstacle ship navigation failure reward. Get the termination status
    r_obs_ship_nav_failure, termination_5 = obs_ship_nav_failure_reward(obs_e_ct, 
                                                                        current_step_accumulated_rewards,
                                                                        is_obs_nav_failure, 
                                                                        e_ct_threshold=obs_e_ct_threshold)
    
    # Evaluate all the non-terminal reward function first
    current_acc_reward_list = np.array([r_ship_collision, r_test_ship_grounding, 
                                        r_test_ship_nav_failure, r_obs_ship_grounding, 
                                        r_obs_ship_nav_failure])
    if normalize_reward:
        normalizing_factor = len(current_acc_reward_list)
    else:
        normalizing_factor = 1 # Not normalized
    current_acc_reward = np.sum(current_acc_reward_list) / normalizing_factor
    
    # GET THE TOTAL REWARD FOR ONE STEP
    r_total = current_acc_reward
    
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
    
    # Track the reward evolution
    reward_log.update(r_ship_collision / normalizing_factor,
                      r_test_ship_grounding / normalizing_factor,
                      r_test_ship_nav_failure / normalizing_factor,
                      r_obs_ship_grounding / normalizing_factor,
                      r_obs_ship_nav_failure / normalizing_factor,
                      r_total)
    
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
    
    return r_total, env_info
    

def ships_collision_reward(test_to_obs_distance, current_step_accumulated_rewards, encounter_type, is_collision, termination_multiplier=10.0):
    '''
    * ships_distance is ALWAYS a positive value
    The reward design consists of two evaluation functions:
    - Reward Design 4 for handling encounter reward
        + The goal is to increase the reward when the both ship is getting closer (within 10 km) 
          and in a head-on or crossing condition
    - Another Reward Design 4 for handing overtake reward:
        + The goal is to still reward the agent because the ship still in proximity, but it decays much 
          faster (reward = 0 when the ship distance is below 200 m)
    - When the ship collide, implement the termination multiplier for the terminal reward
    
    Note: Reward Design parameter is obtained by self tune process
    '''
    # Initiate reward designs, termination status, trigger distance, and initial reward
    # encounter_reward = RewardDesign4(target=0, offset_param=1500000)
    encounter_reward = RewardDesign4(target=0, offset_param=200000002)
    overtake_reward  = RewardDesign4(target=0, offset_param=5000)
    termination = False
    trigger_distance = 10000
    release_distance = 200
    reward = 0
    
    # Compute reward
    if test_to_obs_distance < trigger_distance:
        # If the ships in head-on or side by side condition
        if encounter_type in ["head-on", "crossing"]:
            reward = encounter_reward(test_to_obs_distance)
        
        # If the ship under test overtake the ships
        elif encounter_type == "overtake" and (test_to_obs_distance < release_distance):
            reward = overtake_reward(test_to_obs_distance)
            
    # If is_collision, compute terminal reward
    if is_collision:
        if current_step_accumulated_rewards > 0:
            reward = current_step_accumulated_rewards * termination_multiplier
        elif current_step_accumulated_rewards < 0:
            reward = -current_step_accumulated_rewards * termination_multiplier
        termination = True
            
    return reward, termination

def test_ship_grounding_reward(test_to_ground_distance, current_step_accumulated_rewards, is_test_grounding, termination_multiplier=5.0):
    '''
    * ship_ground_distance is ALWAYS a positive value
    The reward design is based on Reward Design 4:
    - The reward is range between [0, 1]
    - We get a reward closer to 1 when the ship_ground_distance is closer to 0
    - The greater ship_ground_distance is, the lower the reward
    - As the ship_ground_distance is greater than 1 km, reward = 0
    - Terminate the simulation if the obstacle ship experiences grounding
    - Ship under test needs to be within clipping distance from the land to activate the reward function
    
    Note: Reward Design parameter is obtained by self tune process
    '''
    # Initiate reward designs, termination status, clipping distance, and initial reward
    # base_reward = RewardDesign4(target=0, offset_param=750000)      # For distance threshold of 2km
    # base_reward = RewardDesign4(target=0, offset_param=500000)      # For distance threshold of 1.5km
    base_reward = RewardDesign4(target=0, offset_param=175000)      # For distance threshold of 1km
    # base_reward = RewardDesign4(target=0, offset_param=50000)       # For distance threshold of 0.5km
    # base_reward = RewardDesign4(target=0, offset_param=12500)       # For distance threshold of 0.25km
    termination = False
    clipping_distance = 1000
    reward = 0
    
    # Compute reward
    if test_to_ground_distance <= clipping_distance:
        reward = base_reward(test_to_ground_distance)
    elif test_to_ground_distance > clipping_distance:
        reward = 0
    
    # If is_grounding, compute terminal reward
    if is_test_grounding:
        if current_step_accumulated_rewards > 0:
            reward = current_step_accumulated_rewards * termination_multiplier
        elif current_step_accumulated_rewards < 0:
            reward = -current_step_accumulated_rewards * termination_multiplier
        termination = True

    return reward, termination

def test_ship_nav_failure_reward(test_e_ct, current_step_accumulated_rewards, is_test_nav_failure, e_ct_threshold = 3000, termination_multiplier=5.0):
    '''
    * Cross track error is ALWAYS a positive value
    The reward design is based on Reward Design 3:
    - The reward is range between [0, 1]
    - The reward is starts becoming significant when the cross track error goes above 0.5km m
    - We get a reward closer to 1 when the cross track error is closer to 3 km
    - The greater cross track error is, the higher the reward
    - Navigation failure happens when the ship under test cross track error goes beyond 3 km
    - When navigation failure happens, terminate the simulator
    
    Note: Reward Design parameter is obtained by self tune process
    '''
    # Initiate cross track error threshold, reward designs, termination status, and initial reward
    base_reward = RewardDesign3(target=e_ct_threshold, offset_param=1250000)
    termination = False
    reward = 0
    
    # Compute reward
    reward = base_reward(np.abs(test_e_ct))
    
    # If is_nav_failure
    if is_test_nav_failure:
        if current_step_accumulated_rewards > 0:
            reward = current_step_accumulated_rewards * termination_multiplier
        elif current_step_accumulated_rewards < 0:
            reward = -current_step_accumulated_rewards * termination_multiplier
        termination = True
    
    return reward, termination

def obs_ship_grounding_reward(obs_to_ground_distance, current_step_accumulated_rewards, is_obs_grounding, termination_multiplier=2.5):
    '''
    * ship_ground_distance is ALWAYS a positive value
    The reward design is based on Reward Design 4:
    - The reward is range between [-1, 0]
    - We get a reward closer to -1 when the ship_ground_distance is closer to 0
    - The greater ship_ground_distance is, the bigger the reward
    - As the ship_ground_distance is greater than 1 km, reward = 0
    - Terminate the simulation if the obstacle ship experiences grounding
    - Obstacle ship needs to be within clipping distance from the land to activate the reward function
    
    Note: Reward Design parameter is obtained by self tune process
    '''
    # Initiate reward designs, termination status, clipping distance, and initial reward
    # base_reward = RewardDesign4(target=0, offset_param=750000)      # For distance threshold of 2km
    # base_reward = RewardDesign4(target=0, offset_param=500000)      # For distance threshold of 1.5km
    # base_reward = RewardDesign4(target=0, offset_param=175000)      # For distance threshold of 1km
    base_reward = RewardDesign4(target=0, offset_param=50000)       # For distance threshold of 0.5km
    # base_reward = RewardDesign4(target=0, offset_param=12500)       # For distance threshold of 0.25km
    termination = False
    clipping_distance = 1000
    reward = 0
    
    # Compute reward (negative for obstacle ship)
    if obs_to_ground_distance <= clipping_distance:
        reward = -base_reward(obs_to_ground_distance) * 50
    elif obs_to_ground_distance > clipping_distance:
        reward = 0
    
    # If is_grounding, compute terminal reward
    if is_obs_grounding:
        if current_step_accumulated_rewards > 0:
            reward = -current_step_accumulated_rewards * termination_multiplier
        elif current_step_accumulated_rewards < 0:
            reward = current_step_accumulated_rewards * termination_multiplier
        termination = True

    return reward, termination

def obs_ship_nav_failure_reward(obs_e_ct, current_step_accumulated_rewards, is_obs_nav_failure, e_ct_threshold = 500, termination_multiplier=2.5):
    '''
    * Cross track error is ALWAYS a positive value
    The reward design is based on Reward Design 3:
    - The reward is range between [-1, 0]
    - The reward is starts becoming significant when the cross track error goes above 250 m
    - We get a reward closer to -1 when the cross track error is closer to 500 m
    - The greater cross track error is, the higher the reward
    - Navigation failure happens when the obstacle ship failed to reach the next waypoint after 
      traversing for more than (tolerance_coeff * AB_segment_length) distance.
    - When navigation failure happens, terminate the simulator
    
    Note: Reward Design parameter is obtained by self tune process
    '''
    # Initiate cross track error threshold, reward designs and termination status
    base_reward = RewardDesign3(target=e_ct_threshold, offset_param=12500)
    termination = False
    
    # Compute reward
    reward = -base_reward(np.abs(obs_e_ct))
    
    # If is_nav_failure
    if is_obs_nav_failure:
        if current_step_accumulated_rewards > 0:
            reward = -current_step_accumulated_rewards * termination_multiplier
        elif current_step_accumulated_rewards < 0:
            reward = current_step_accumulated_rewards * termination_multiplier
        termination = True
    
    return reward, termination

def obs_ship_IW_sampling_failure_reward(current_step_accumulated_rewards, is_sampling_failure, termination_multiplier=5):
    '''
    Compute reward when obstacle ship sample the intermediate waypoint in the terrain.
    
    If:
    - The current accumulated reward is positive, halv the reward as a penalty
    - The current accumulated reward is negative, inverse-halv (double) the reward as penalty
    
    This is done to encourage the agent not to sample on a terrain, but not necessarily diminish its learning effort when
    when it did a good sampling before (by making the reward negative when current accumulated reward is positive). But punish 
    it more when the agent did not perform well beforehand (by keeping it negative and multiplyng by two when current accumulated reward is negative)
    '''
    # Initiate the reward and termination status
    reward = current_step_accumulated_rewards
    termination = False
    
    # Compute terminal reward
    if is_sampling_failure:
        # When current_acc_reward is positive, we HALV THE ACCUMULATED REWARD IN WHOLE EPISODE, 
        # then deducted it (which already has positive value) from the total accumulated reward
        if current_step_accumulated_rewards >= 0:
            reward = -reward * termination_multiplier
        # When current_acc_reward is positive, we HALV THE ACCUMULATED REWARD IN WHOLE EPISODE, 
        # then add it (which already has negative value) from the total accumulated reward
        elif current_step_accumulated_rewards < 0:
            reward = reward * termination_multiplier
        termination = True
        
    return reward, termination