import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from utils.center_plot import center_plot_window
from utils.animate import RLShipTrajectoryAnimator
import os

class ShowPostTrainedEnvResult():
    def __init__(self, env):
        '''
        Takes post-trained wrapped environment, then plot and animate the results
        '''
        self.expl_env = env
        
        self.test = self.expl_env.wrapped_env.test
        self.obs = self.expl_env.wrapped_env.obs
        
        # Get the simulation results for all assets
        self.ts_results_df = pd.DataFrame().from_dict(self.expl_env.wrapped_env.test.ship_model.simulation_results)
        self.os_results_df = pd.DataFrame().from_dict(self.expl_env.wrapped_env.obs.ship_model.simulation_results)
        
    def do_plot_and_animate(self,
                            plot_1=False,
                            plot_2=False,
                            plot_3=False,
                            plot_4=False,
                            animation=False):
        if animation:
            test_route = {'east': self.test.auto_pilot.navigate.east, 'north': self.test.auto_pilot.navigate.north}
            obs_route = {'east': self.obs.auto_pilot.navigate.east, 'north': self.obs.auto_pilot.navigate.north}
            waypoint_sampling_times = self.expl_env.wrapped_env.waypoint_sampling_times
            waypoints = list(zip(self.obs.auto_pilot.navigate.east, self.obs.auto_pilot.navigate.north))
            map_obj = self.expl_env.wrapped_env.map
            radius_of_acceptance = self.expl_env.wrapped_env.args.radius_of_acceptance
            timestamps = self.test.time_list
            interval = self.expl_env.wrapped_env.args.time_step * 1000
            test_drawer=self.test.ship_model.draw
            obs_drawer=self.obs.ship_model.draw
            test_headings=self.ts_results_df["yaw angle [deg]"]
            obs_headings=self.os_results_df["yaw angle [deg]"]
            is_collision_imminent_list = self.expl_env.wrapped_env.is_collision_imminent_list
            is_collision_list = self.expl_env.wrapped_env.is_collision_list
            
            # Do animation
            animator = RLShipTrajectoryAnimator(self.ts_results_df,
                                                self.os_results_df,
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
            fps = 480
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
                self.expl_env.wrapped_env.reward_tracker.test_ship_grounding,
                self.expl_env.wrapped_env.reward_tracker.test_ship_nav_failure,
                self.expl_env.wrapped_env.reward_tracker.from_test_ship,
                self.expl_env.wrapped_env.reward_tracker.ship_collision,
                self.expl_env.wrapped_env.reward_tracker.obs_ship_grounding,
                self.expl_env.wrapped_env.reward_tracker.obs_ship_nav_failure,
                self.expl_env.wrapped_env.reward_tracker.from_obs_ship,
                self.expl_env.wrapped_env.reward_tracker.total,
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
                self.expl_env.wrapped_env.reward_tracker.test_ship_grounding,
                self.expl_env.wrapped_env.reward_tracker.test_ship_nav_failure,
                self.expl_env.wrapped_env.reward_tracker.from_test_ship,
                self.expl_env.wrapped_env.reward_tracker.ship_collision,
                self.expl_env.wrapped_env.reward_tracker.obs_ship_grounding,
                self.expl_env.wrapped_env.reward_tracker.obs_ship_nav_failure,
                self.expl_env.wrapped_env.reward_tracker.from_obs_ship,
                self.expl_env.wrapped_env.reward_tracker.total,
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
            axes[0].plot(self.ts_results_df['time [s]'], self.ts_results_df['forward speed [m/s]'])
            axes[0].axhline(y=self.test.desired_forward_speed, color='red', linestyle='--', linewidth=1.5, label='Desired Forward Speed')
            axes[0].set_title('Test Ship Forward Speed [m/s]')
            axes[0].set_xlabel('Time (s)')
            axes[0].set_ylabel('Forward Speed (m/s)')
            axes[0].grid(color='0.8', linestyle='-', linewidth=0.5)
            axes[0].set_xlim(left=0)

            # Plot 3.1: Forward Speed
            axes[1].plot(self.os_results_df['time [s]'], self.os_results_df['forward speed [m/s]'])
            axes[1].axhline(y=self.obs.desired_forward_speed, color='red', linestyle='--', linewidth=1.5, label='Desired Forward Speed')
            axes[1].set_title('Obstacle Ship Forward Speed [m/s]')
            axes[1].set_xlabel('Time (s)')
            axes[1].set_ylabel('Forward Speed (m/s)')
            axes[1].grid(color='0.8', linestyle='-', linewidth=0.5)
            axes[1].set_xlim(left=0)

            # Plot 2.2: Rudder Angle
            axes[2].plot(self.ts_results_df['time [s]'], self.ts_results_df['rudder angle [deg]'])
            axes[2].set_title('Test Ship Rudder angle [deg]')
            axes[2].set_xlabel('Time (s)')
            axes[2].set_ylabel('Rudder angle [deg]')
            axes[2].grid(color='0.8', linestyle='-', linewidth=0.5)
            axes[2].set_xlim(left=0)
            axes[2].set_ylim(-31,31)

            # Plot 3.2: Rudder Angle
            axes[3].plot(self.os_results_df['time [s]'], self.os_results_df['rudder angle [deg]'])
            axes[3].set_title('Obstacle Ship Rudder angle [deg]')
            axes[3].set_xlabel('Time (s)')
            axes[3].set_ylabel('Rudder angle [deg]')
            axes[3].grid(color='0.8', linestyle='-', linewidth=0.5)
            axes[3].set_xlim(left=0)
            axes[3].set_ylim(-31,31)

            # Plot 2.3: Cross Track error
            axes[4].plot(self.ts_results_df['time [s]'], self.ts_results_df['cross track error [m]'])
            axes[4].set_title('Test Ship Cross Track Error [m]')
            axes[4].set_xlabel('Time (s)')
            axes[4].set_ylabel('Cross track error (m)')
            axes[4].grid(color='0.8', linestyle='-', linewidth=0.5)
            axes[4].set_xlim(left=0)

            # Plot 3.3: Cross Track error
            axes[5].plot(self.os_results_df['time [s]'], self.os_results_df['cross track error [m]'])
            axes[5].set_title('Obstacle Ship Cross Track Error [m]')
            axes[5].set_xlabel('Time (s)')
            axes[5].set_ylabel('Cross track error (m)')
            axes[5].grid(color='0.8', linestyle='-', linewidth=0.5)
            axes[5].set_xlim(left=0)

            # Plot 2.4: Propeller Shaft Speed
            axes[6].plot(self.ts_results_df['time [s]'], self.ts_results_df['propeller shaft speed [rpm]'])
            axes[6].set_title('Test Ship Propeller Shaft Speed [rpm]')
            axes[6].set_xlabel('Time (s)')
            axes[6].set_ylabel('Propeller Shaft Speed (rpm)')
            axes[6].grid(color='0.8', linestyle='-', linewidth=0.5)
            axes[6].set_xlim(left=0)

            # Plot 3.4: Propeller Shaft Speed
            axes[7].plot(self.os_results_df['time [s]'], self.os_results_df['propeller shaft speed [rpm]'])
            axes[7].set_title('Obstacle Ship Propeller Shaft Speed [rpm]')
            axes[7].set_xlabel('Time (s)')
            axes[7].set_ylabel('Propeller Shaft Speed (rpm)')
            axes[7].grid(color='0.8', linestyle='-', linewidth=0.5)
            axes[7].set_xlim(left=0)

            # Plot 2.5: Power vs Available Power
            axes[8].plot(self.ts_results_df['time [s]'], self.ts_results_df['power electrical [kw]'], label="Power")
            axes[8].plot(self.ts_results_df['time [s]'], self.ts_results_df['available power electrical [kw]'], label="Available Power")
            axes[8].set_title('Test Ship Power vs Available Power [kw]')
            axes[8].set_xlabel('Time (s)')
            axes[8].set_ylabel('Power (kw)')
            axes[8].legend()
            axes[8].grid(color='0.8', linestyle='-', linewidth=0.5)
            axes[8].set_xlim(left=0)

            # Plot 3.5: Power vs Available Power
            axes[9].plot(self.os_results_df['time [s]'], self.os_results_df['power electrical [kw]'], label="Power")
            axes[9].plot(self.os_results_df['time [s]'], self.os_results_df['available power electrical [kw]'], label="Available Power")
            axes[9].set_title('Obstacle Ship Power vs Available Power [kw]')
            axes[9].set_xlabel('Time (s)')
            axes[9].set_ylabel('Power (kw)')
            axes[9].legend()
            axes[9].grid(color='0.8', linestyle='-', linewidth=0.5)
            axes[9].set_xlim(left=0)

            # Plot 2.6: Fuel Consumption
            axes[10].plot(self.ts_results_df['time [s]'], self.ts_results_df['fuel consumption [kg]'])
            axes[10].set_title('Test Ship Fuel Consumption [kg]')
            axes[10].set_xlabel('Time (s)')
            axes[10].set_ylabel('Fuel Consumption (kg)')
            axes[10].grid(color='0.8', linestyle='-', linewidth=0.5)
            axes[10].set_xlim(left=0)

            # Plot 3.6: Fuel Consumption
            axes[11].plot(self.os_results_df['time [s]'], self.os_results_df['fuel consumption [kg]'])
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
            plt.plot(self.ts_results_df['east position [m]'].to_numpy(), self.ts_results_df['north position [m]'].to_numpy())
            plt.scatter(self.test.auto_pilot.navigate.east, self.test.auto_pilot.navigate.north, marker='x', color='blue')  # Waypoints
            plt.plot(self.test.auto_pilot.navigate.east, self.test.auto_pilot.navigate.north, linestyle='--', color='blue')  # Line
            for x, y in zip(self.test.ship_model.ship_drawings[1], self.test.ship_model.ship_drawings[0]):
                plt.plot(x, y, color='blue')
            # for i, (east, north) in enumerate(zip(test.auto_pilot.navigate.east, test.auto_pilot.navigate.north)):
            #     test_radius_circle = Circle((east, north), args.radius_of_acceptance, color='blue', alpha=0.3, fill=True)
            #     plt.gca().add_patch(test_radius_circle)
            # Obs ship    
            plt.plot(self.os_results_df['east position [m]'].to_numpy(), self.os_results_df['north position [m]'].to_numpy())
            plt.scatter(self.obs.auto_pilot.navigate.east, self.obs.auto_pilot.navigate.north, marker='x', color='red')  # Waypoints
            plt.plot(self.obs.auto_pilot.navigate.east, self.obs.auto_pilot.navigate.north, linestyle='--', color='red')  # Line
            for x, y in zip(self.obs.ship_model.ship_drawings[1], self.obs.ship_model.ship_drawings[0]):
                plt.plot(x, y, color='red')
            for i, (east, north) in enumerate(zip(self.obs.auto_pilot.navigate.east, self.obs.auto_pilot.navigate.north)):
                obs_radius_circle = Circle((east, north), self.expl_env.args.radius_of_acceptance, color='red', alpha=0.3, fill=True)
                plt.gca().add_patch(obs_radius_circle)
            self.expl_env.wrapped_env.map.plot_obstacle(plt.gca())  # get current Axes to pass into map function

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