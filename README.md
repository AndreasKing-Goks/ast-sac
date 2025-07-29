# ast-sac
Repository for **Adaptive Stress Testing** using the **Soft Actor-Critic (SAC)** algorithm in a ship transit simulation environment.

## Conda Environment Setup

To set up the environment, run:

```bash
conda env create -f ast-sac.yml
```

> Note: You may need to downgrade `numpy` to a version `<2.0` due to compatibility issues with TensorFlow.

---

##  Using the Ship in Transit Simulator

Text

---

## Training the AST-SAC Agent

Training the agent involves four key steps:

### 1. Configure the Environment and Agent (in `ast-sac_runner.py`)

All training and environment settings are defined using `argparse`. Below is an overview:

#### Environment Arguments:
| Argument | Description |
|----------|-------------|
| `--max_sampling_frequency` | Max action sampling steps per episode (default: 9) |
| `--time_step` | Simulator time step size in seconds (default: 4) |
| `--radius_of_acceptance` | Radius threshold for LOS guidance (default: 300) |
| `--lookahead_distance` | Lookahead distance for LOS (default: 1000) |
| `--collav_mode` | Collision avoidance mode: `["none", "simple", "sbmpc"]` (default: "sbmpc") |
| `--ship_draw` | Enable ship drawing for animation (default: True) |
| `--time_since_last_ship_drawing` | Drawing delay in seconds (default: 30) |
| `--normalize_action` | Normalize environment action space (default: False) |

#### SAC Algorithm Arguments:
| Argument | Description |
|----------|-------------|
| `--do_logging` | Enable training logging (default: True) |
| `--algorithm` | RL algorithm for AST (default: "SAC") |
| `--version` | Version label (default: "normal") |
| `--layer_size` | Neural net hidden layer size (default: 256) |
| `--replay_buffer_size` | Size of replay buffer (default: 300000) |
| `--batch_size` | Batch size for updates (default: 256) |
| `--num_epochs` | Training epochs (default: 500) |
| `--num_eval_steps_per_epoch` | Evaluation steps per epoch (default: 180) |
| `--num_trains_per_train_loop` | Gradient updates per training loop (default: 240) |
| `--num_expl_steps_per_train_loop` | Exploration steps per loop (default: 256) |
| `--min_num_steps_before_training` | Initial steps before training (default: 8192) |
| `--max_path_length` | Max steps per episode (default: 9) |

#### SAC Trainer Arguments:
| Argument | Description |
|----------|-------------|
| `--discount` | Discount factor (default: 0.965) |
| `--soft_target_tau` | Soft target update rate (default: 1e-3) |
| `--target_update_period` | Target net update interval (default: 1) |
| `--policy_lr` | Learning rate for policy net (default: 8e-5) |
| `--qf_lr` | Learning rate for Q-net (default: 8e-5) |
| `--reward_scale` | Reward scaling (default: 0.75) |
| `--use_automatic_entropy_tuning` | Enable auto-entropy tuning (default: True) |
| `--action_reg_coeff` | Coefficient for action regularization (default: 0.01) |
| `--clip_val` | Q-value clipping to avoid instability (default: 100) |

---

### 2. Output Files

After training:
- The trained weights are saved in:  
  `run/logs/<test_name>/<test_unique_id>/params.pkl`
- The training log (performance plots) is saved in:  
  `run/logs/<test_name>/<test_unique_id>/progress.csv`

---

### 3. Simulate the Trained Policy

To visualize a trained policy:

1. Open `ast-sac_run_trained_policy.py`.
2. Locate line 433 and update the `trained` variable to the directory containing the `params.pkl` file.
3. The simulation will show an animation by default.
   - To disable animation: set `animate=False` in the `animate()` function.
   - To save animation: set `save_animation=True` and specify the path via `ani_dir` in the `animate()` function.

---

### 4. Visualize Training Progress

To visualize the reward and training curve:

1. Open `ast-sac_progress_plotter.py`.
2. Update the `csv_path` variable (line 94) to the path of your desired `progress.csv` file.

---


