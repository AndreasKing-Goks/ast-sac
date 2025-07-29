# ast-sac
Repository for **Adaptive Stress Testing** using the **Soft Actor-Critic (SAC)** algorithm in a ship transit simulation environment.

## Conda Environment Setup

To set up the environment, run:

```bash
conda env create -f ast-sac.yml
```

> Note: You may need to downgrade `numpy` to a version `<2.0` due to compatibility issues with TensorFlow.

---

##  Ship in Transit Simulator

The **ship-in-transit-simulator** is a modular Python-based simulation framework for modeling and running transit scenarios of a marine vessel. It includes ship dynamics, machinery system behaviors, navigation logic, and environmental effects. 

This simulator is developed by Børge Rokseth (**borge.rokseth@ntnu.no**). Original simulator can be found at: https://github.com/BorgeRokseth/ship_in_transit_simulator.git


### Ship Dynamics

The ship model simulates motion in **three degrees of freedom**:

- **Surge**
- **Sway**
- **Yaw**

#### State Variables

The full system state includes:

- `x_N`: North position [m]
- `y_E`: East position [m]
- `ψ`: Yaw angle [rad]
- `u`: Surge velocity [m/s]
- `v`: Sway velocity [m/s]
- `r`: Yaw rate (turn rate) [rad/s]

**Additional states** (depending on machinery system model):

- `ω_prop`: Propeller shaft angular velocity [rad/s] — for detailed machinery system
- `T`: Thrust force [N] — for simplified model

#### Forces Modeled

- Inertial forces
- Added mass effects
- Coriolis forces
- Linear and nonlinear damping
- Environmental forces (wind, current)
- Control forces (propeller & rudder)

### Machinery System

The ship includes:
- **1 propeller shaft**
- **1 rudder**
- Powered by either a **main diesel engine** or a **hybrid shaft generator**.

#### Energy Sources

- **Main Engine (ME)**: Primary mechanical power source
- **Hybrid Shaft Generator (HSG)**: Can operate as motor/generator
- **Electrical Distribution**: Powered by diesel generators

#### Machinery Modes

| Mode   | Propulsion Power      | Hotel Load Power       |
|--------|------------------------|-------------------------|
| PTO    | Main Engine            | Hybrid SG (generator)   |
| MEC    | Main Engine            | Electrical Distribution |
| PTI    | Hybrid SG (motor)      | Electrical Distribution |

#### Available Machinery Models

1. **Detailed Model** — includes propeller shaft dynamics
2. **Simplified Model** — thrust modeled as a 2nd order transfer function

### Navigation System

Two navigation modes:

- **Heading-by-Reference**: Maintains a specified heading
- **Waypoint Controller**: Follows a sequence of waypoints

### Environmental Forces

- **Wind**: Constant wind speed and direction
- **Current**: Constant current velocity

### Speed Control Options

1. **Throttle Input**: A direct command representing propulsion load percentage
2. **Speed Controller**: Regulates propeller shaft speed to maintain desired vessel speed

###  Example Scenarios

- Single-engine configuration using only the Main Engine in PTO mode
- Complex hybrid-electric propulsion control

For usage and integration examples, refer to the provided scripts and documentation in `simu_standalone/`

---

## Setting up the Ship in Transit Simulator

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


