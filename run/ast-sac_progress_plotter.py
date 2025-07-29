from utils.RL_train_progress_plotter import ProgressPlotter

'''
Available columns:
- Epoch
- epoch
- eval/Actions Max
- eval/Actions Mean
- eval/Actions Min
- eval/Actions Std
- eval/Average Returns
- eval/Num Paths
- eval/Returns Max
- eval/Returns Mean
- eval/Returns Min
- eval/Returns Std
- eval/Rewards Max
- eval/Rewards Mean
- eval/Rewards Min
- eval/Rewards Std
- eval/num paths total
- eval/num steps total
- eval/path length Max
- eval/path length Mean
- eval/path length Min
- eval/path length Std
- expl/Actions Max
- expl/Actions Mean
- expl/Actions Min
- expl/Actions Std
- expl/Average Returns
- expl/Num Paths
- expl/Returns Max
- expl/Returns Mean
- expl/Returns Min
- expl/Returns Std
- expl/Rewards Max
- expl/Rewards Mean
- expl/Rewards Min
- expl/Rewards Std
- expl/num paths total
- expl/num steps total
- expl/path length Max
- expl/path length Mean
- expl/path length Min
- expl/path length Std
- replay_buffer/size
- time/data storing (s)
- time/epoch (s)
- time/evaluation sampling (s)
- time/exploration sampling (s)
- time/logging (s)
- time/sac training (s)
- time/saving (s)
- time/total (s)
- time/training (s)
- trainer/Alpha
- trainer/Alpha Loss
- trainer/Log Pis Max
- trainer/Log Pis Mean
- trainer/Log Pis Min
- trainer/Log Pis Std
- trainer/Policy Loss
- trainer/Q Targets Max
- trainer/Q Targets Mean
- trainer/Q Targets Min
- trainer/Q Targets Std
- trainer/Q1 Predictions Max
- trainer/Q1 Predictions Mean
- trainer/Q1 Predictions Min
- trainer/Q1 Predictions Std
- trainer/Q2 Predictions Max
- trainer/Q2 Predictions Mean
- trainer/Q2 Predictions Min
- trainer/Q2 Predictions Std
- trainer/QF1 Loss
- trainer/QF2 Loss
- trainer/num train calls
- trainer/policy/mean Max
- trainer/policy/mean Mean
- trainer/policy/mean Min
- trainer/policy/mean Std
- trainer/policy/normal/log_std Max
- trainer/policy/normal/log_std Mean
- trainer/policy/normal/log_std Min
- trainer/policy/normal/log_std Std
- trainer/policy/normal/std Max
- trainer/policy/normal/std Mean
- trainer/policy/normal/std Min
- trainer/policy/normal/std Std
'''

# Get the progress.csv path
csv_path = r'D:\OneDrive - NTNU\PhD\PhD_Projects\ast-sac\run\logs\ast-sac-maritime-logs\ast-sac_maritime_logs_2025_07_15_22_46_15_0000--s-0\progress.csv'

# Instantiate the plotter
plotter = ProgressPlotter(csv_path)

## Learning Stability & Divergence Detection
# Plot the Q-function Losses
plot_1 = True
# plot_1 = False
if plot_1:
    plotter.plot_column('trainer/QF1 Loss')
    plotter.plot_column('trainer/QF2 Loss')

# Plot the Q prediction and targets
plot_2 = True
# plot_2 = False
if plot_2:
    plotter.plot_columns(['trainer/Q1 Predictions Mean', 'trainer/Q1 Predictions Min', 'trainer/Q1 Predictions Max'])
    plotter.plot_columns(['trainer/Q2 Predictions Mean', 'trainer/Q2 Predictions Min', 'trainer/Q2 Predictions Max'])
    plotter.plot_columns(['trainer/Q Targets Mean', 'trainer/Q Targets Min', 'trainer/Q Targets Max'])

# Policy Loss and Entropy
plot_3 = True
# plot_3 = False
if plot_3:
    plotter.plot_column('trainer/Policy Loss')
    plotter.plot_columns(['trainer/Log Pis Mean', 'trainer/Log Pis Min', 'trainer/Log Pis Max'])
    plotter.plot_column('trainer/Alpha')
    plotter.plot_column('trainer/Alpha Loss')

## Policy Behavior
# Policy Outputs Stats
plot_4 = True
# plot_4 = False
if plot_4:
    plotter.plot_columns(['trainer/policy/mean Mean', 'trainer/policy/mean Std', 'trainer/policy/mean Min', 'trainer/policy/mean Max'])
    plotter.plot_columns(['trainer/policy/normal/std Mean', 'trainer/policy/normal/log_std Mean'])

# Exploration and Evaluation Actions
plot_5 = True
# plot_5 = False
if plot_5:
    plotter.plot_columns(['expl/Actions Mean', 'expl/Actions Std', 'expl/Actions Min', 'expl/Actions Max'])
    plotter.plot_columns(['eval/Actions Mean', 'eval/Actions Std', 'eval/Actions Min', 'eval/Actions Max'])

## Reward and Return Evaluation
# Reward and Returns (Exploration)
plot_6 = True
# plot_6 = False
if plot_6:
    plotter.plot_columns(['expl/Rewards Mean', 'expl/Rewards Max', 'expl/Rewards Min'])
    plotter.plot_columns(['expl/Returns Mean', 'expl/Returns Max', 'expl/Returns Min'])

# Rewards and Returns (Evaluation)
plot_7 = True
# plot_7 = False
if plot_7:
    plotter.plot_columns(['eval/Rewards Mean', 'eval/Rewards Max', 'eval/Rewards Min'])
    plotter.plot_column('eval/Returns Mean')

### Time Profiling
plot_8 = True
# plot_8 = False
if plot_8:
    plotter.plot_column('trainer/Policy Loss')
    plotter.plot_column('time/sac training (s)')
    plotter.plot_column('time/exploration sampling (s)')
    plotter.plot_column('time/evaluation sampling (s)')
    plotter.plot_column('time/logging (s)')
    plotter.plot_column('time/epoch (s)')

plotter.plot_column_paper('eval/Returns Mean')

# Show Plot
plotter.show_plot()


