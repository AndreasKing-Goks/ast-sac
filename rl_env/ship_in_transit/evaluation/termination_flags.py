import numpy as np
from rl_env.ship_in_transit.sub_systems.obstacle import PolygonObstacle
from rl_env.ship_in_transit.evaluation import check_condition

def get_termination_status(states,
                           test_route_end,
                           obs_route_end,
                           map_obj:PolygonObstacle,
                           test_ship_length,
                           obs_ship_length):
        
    # Initial termination status
    termination = False
        
    # Unpack states
    test_pos    = states[0:2]
    test_e_ct   = states[2]
    obs_pos     = states[3:5]
    obs_heading = states[5]
    obs_speed   = states[6]
    obs_e_ct    = states[7]
        
    ## Do evaluation function
    # Check if ship under test reaches end point
    test_is_reached = check_condition.is_reaches_endpoint(test_route_end, 
                                                          test_pos)
    
    # Check if ship under test goes outside map horizon
    test_is_outside = check_condition.is_pos_outside_horizon(map_obj,
                                                             test_pos,
                                                             test_ship_length)
    
    # Check if ship under test hit the ground
    test_is_grounded = check_condition.is_pos_inside_obstacles(map_obj,
                                                               test_pos,
                                                               test_ship_length)
    
    # Check if ship under test encounters navigation failure
    test_is_failed_nav = check_condition.is_ship_navigation_failure(test_e_ct)
    
    # Check if ship under test reaches end point
    obs_is_reached = check_condition.is_reaches_endpoint(obs_route_end, 
                                                         obs_pos)
    
    # Check if obstacle ship goes outside map horizon
    obs_is_outside = check_condition.is_pos_outside_horizon(map_obj,
                                                            obs_pos,
                                                            obs_ship_length)
    
    # Check if obstacle ship hit the ground
    obs_is_grounded = check_condition.is_pos_inside_obstacles(map_obj,
                                                              obs_pos,
                                                              obs_ship_length)
    
    # Check if obstacle ship encounters navigation failure
    obs_is_failed_nav = check_condition.is_ship_navigation_failure(obs_e_ct)
    
    # Check if all ships nearing collision
    is_near_collision = check_condition.is_collision_imminent(test_pos,
                                                              obs_pos)
    
    # Check if ship collision happens
    is_collision = check_condition.is_ship_collision(test_pos,
                                                     obs_pos)
    
    termination_conditions = [test_is_reached, test_is_outside, test_is_grounded, test_is_failed_nav, 
                              is_near_collision, is_collision, 
                              obs_is_reached, obs_is_outside, obs_is_grounded, obs_is_failed_nav]
    
    return termination_conditions