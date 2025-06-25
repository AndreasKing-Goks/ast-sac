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
from rl_env.ship_in_transit.reward_functions.check_condition import (is_reaches_endpoint,
                                                                     is_pos_outside_horizon,
                                                                     is_pos_inside_obstacles,
                                                                     is_ship_navigation_failure,
                                                                     is_route_inside_obstacles,
                                                                     is_route_outside_horizon,
                                                                     is_collision_imminent,
                                                                     is_ship_collision)
from rl_env.ship_in_transit.utils.compute_distance import get_euclidean_distance

def ship_collision_reward(ships_distance, encounter_type, is_collision, termination_multiplier=100.0):
    '''
    * ships_distance is ALWAYS a positive value
    The reward design consists of two evaluation functions:
    - Reward Design 4 for handling encounter reward
        + The goal is to increase the reward when the both ship is getting closer (within 3 km) 
          and in a head-on or crossing condition
    - Another Reward Design 4 for handing overtake reward:
        + The goal is to still reward the agent because the ship still in proximity, but it decays much 
          faster (reward = 0 when the ship distance is below 200 m)
    - When the ship collide, implement the termination multiplier for the terminal reward
    
    Note: Reward Design parameter is obtained by self tune process
    '''
    # Initiate reward designs, termination status, trigger distance, and initial reward
    encounter_reward = RewardDesign4(target=0, offset_param=1500000)
    overtake_reward  = RewardDesign4(target=0, offset_param=5000)
    termination = False
    trigger_distance = 3000
    release_distance = 200
    reward = 0
    
    # Compute reward
    if ships_distance < trigger_distance:
        # If the ships in head-on or side by side condition
        if encounter_type in ["head-on", "crossing"]:
            reward = encounter_reward(ships_distance)
        
        # If the ship under test overtake the ships
        elif encounter_type == "overtake" and (ships_distance < release_distance):
            reward = overtake_reward(ships_distance)
            
    # If is_collision, compute terminal reward
    if is_collision:
        reward = reward * termination_multiplier
        termination = True
            
    return reward, termination

def test_ship_grounding_reward(ship_ground_distance, is_test_grounding, termination_multiplier=50.0):
    '''
    * ship_ground_distance is ALWAYS a positive value
    The reward design is based on Reward Design 4:
    - The reward is range between [0, 1]
    - We get a reward closer to 1 when the ship_ground_distance is closer to 0
    - The greater ship_ground_distance is, the lower the reward
    - As the ship_ground_distance is greater than 2 km, reward = 0
    - Terminate the simulation if the obstacle ship experiences grounding
    
    Note: Reward Design parameter is obtained by self tune process
    '''
    # Initiate reward designs, termination status, clipping distance, and initial reward
    base_reward = RewardDesign4(target=0, offset_param=750000)
    termination = False
    clipping_distance = 2000
    reward = 0
    
    # Compute reward
    if ship_ground_distance <= clipping_distance:
        reward = base_reward(ship_ground_distance)
    elif ship_ground_distance > clipping_distance:
        reward = 0
    
    # If is_grounding, compute terminal reward
    if is_test_grounding:
        reward = reward * termination_multiplier
        termination = True

    return reward, termination

def test_ship_nav_failure_reward(e_ct, is_test_nav_failure, termination_multiplier=25.0):
    '''
    * Cross track error is ALWAYS a positive value
    The reward design is based on Reward Design 3:
    - The reward is range between [0, 1]
    - We get a reward closer to 1 when the cross track error is closer to 1 km
    - The greater cross track error is, the higher the reward
    - Navigation failure happens when the ship failed to reach the next waypoint within a certain time window (TBA)
    - When navigation failure happens, terminate the simulator
    
    Note: Reward Design parameter is obtained by self tune process
    '''
    # Initiate cross track erro threshold, reward designs, termination status, and initial reward
    e_ct_threshold = 1000
    base_reward = RewardDesign3(target=e_ct_threshold, offset_param=150000)
    termination = False
    reward = 0
    
    # Compute reward
    reward = base_reward(e_ct)
    
    # If is_nav_failure
    if is_test_nav_failure:
        reward = reward * termination_multiplier
        termination = True
    
    return reward, termination

def obs_ship_grounding_reward(ship_ground_distance, is_obs_grounding, termination_multiplier=50.0):
    '''
    * ship_ground_distance is ALWAYS a positive value
    The reward design is based on Reward Design 4:
    - The reward is range between [-1, 0]
    - We get a reward closer to -1 when the ship_ground_distance is closer to 0
    - The greater ship_ground_distance is, the bigger the reward
    - As the ship_ground_distance is greater than 2 km m, reward = 0
    - Terminate the simulation if the obstacle ship experiences grounding
    
    Note: Reward Design parameter is obtained by self tune process
    '''
    # Initiate reward designs, termination status, clipping distance, and initial reward
    base_reward = RewardDesign4(target=0, offset_param=750000)
    termination = False
    clipping_distance = 2000
    reward = 0
    
    # Compute reward (negative for obstacle ship)
    if ship_ground_distance <= clipping_distance:
        reward = -base_reward(ship_ground_distance)
    elif ship_ground_distance > clipping_distance:
        reward = 0
    
    # If is_grounding, compute terminal reward
    if is_obs_grounding:
        reward = reward * termination_multiplier
        termination = True

    return reward, termination

def obs_ship_nav_failure_reward(e_ct, is_obs_nav_failure, termination_multiplier=25.0):
    '''
    * Cross track error is ALWAYS a positive value
    The reward design is based on Reward Design 3:
    - The reward is range between [-1, 0]
    - We get a reward closer to -1 when the cross track error is closer to 500 m
    - The greater cross track error is, the higher the reward
    - Navigation failure happens when the obstacle ship cross track error goes beyond 500 m
    - When navigation failure happens, terminate the simulator
    
    Note: Reward Design parameter is obtained by self tune process
    '''
    # Initiate cross track erro threshold, reward designs and termination status
    e_ct_threshold = 500
    base_reward = RewardDesign3(target=e_ct_threshold, offset_param=45000)
    termination = False
    
    # Compute reward
    reward = -base_reward(e_ct)
    
    # If is_nav_failure
    if is_obs_nav_failure:
        reward = reward * termination_multiplier
        termination = True
    
    return reward, termination

def obs_ship_IWsampling_failure_reward(current_acc_reward, is_sampling_failure, termination_multiplier=0.5):
    '''
    Compute reward when obstacle ship sample the intermediate waypoint in the terrain.
    
    If:
    - The current accumulated reward is positive, halv the reward as a penalty
    - The current accumulated reward is negative, inverse-halv (double) the reward as penalty
    '''
    # Initiate the reward and termination status
    reward = current_acc_reward
    termination = False
    
    # Compute terminal reward
    if is_sampling_failure:
        if current_acc_reward >= 0:
            reward = reward * termination_multiplier
        elif current_acc_reward < 0:
            reward = reward / termination_multiplier
        
    return reward