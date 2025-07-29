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

---

## Setting up the Ship in Transit Simulator with RL-learning agent and multi-ship configurations

For usage and integration examples, refer to the provided scripts and documentation in `simu_standalone/multi_ship_rl_env_setup.py`

Generally, building this simulator is done by doing these steps:
1. Prepare configurations objects:
  - Ship configurations built from `ShipConfiguration()`
  - Machinery configurations (including machinery modes) built from `MachinerySystemConfiguration()`
  - Simulation environment configuration built from `EnvironmentConfiguration()`
  - Simulator configuration built from `SimulationConfiguration()`
2. Using this configuration objects, we can then built a ship asset for stress testing purpose using `ShipModelAST()`.
3. The ship model needs throttle controller and heading controller for it to carry an autonomous mission. A simple engine throttle controller with fixed desired speed can be set up using `EngineThrottleFromSpeedSetPoint()`. A heading controller using LOS Guidance can also be set using `HeadingByRouteController()`. For this we need the mission waypoints inside a file named `_ship_route.txt` beforehand.
4. We could also set up a ship model in which it can "choose" a new intermediate waypoint during the simulation. This can be done using `HeadingBySampledRouteController()`. This control is still based on the `HeadingByRouteController()`. The main difference is that the we only need two initial waypoints, start and end point, in the `_ship_route.txt`.
5. We can run multiple ship models simultaneously. In order to do that, we know introduce a new term `ShipAssets()`, where it collects the corresponding controllers and other parameters associated to it. Typicially in stress testing settings, we have a ship asset which undergoes a stress testing, namely `test` ship, and a ship asset that acts as a disturbance to affect the ship under test, namely `obs`. All ship assets is collected inside a list, then used to create the RL environment for the adaptive stress testing.
6. Typically, the first entry in the ship asset list are the `test` ship, and the rest is `obs` ship/s. 
> Multiple obstacle ships implementation is not finished yet!

In `multi_ship_rl_env_setup.py`, we can find 9 tests in total for a better understanding (also used as sanity checks) for the stress-testing algorithm. Below are the descriptions:

| Test No. | Description |
|----------|-------------|
| `test 1` | Test the usage of `normalized_box_env()` function |
| `test 2` | Test a direct action sampling from an RL environment |
| `test 3` | Test the behaviour of `get_intermediate_waypoints()` function |
| `test 4` | Test an action sampling from the policy |
| `test 5` | Test the use of policy's action sampling to step up the simulator |
| `test 6` | Test for simulaion step up using one action sampled from policy |
| `test 7` | Test the simulation step up using policy's action sampling or direct action manipulation |
| `test 8` | Test the step - reset - step process of the simulation |
| `test 9` | Test the rollout function for adaptive stress testing purpose |
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


