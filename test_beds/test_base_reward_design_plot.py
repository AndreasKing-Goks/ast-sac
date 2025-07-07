import matplotlib.pyplot as plt
import numpy as np

from rl_env.reward_designs import (
    RewardDesign1, RewardDesign2, RewardDesign3,
    RewardDesign4, RewardDesign5, RewardDesign6
)

# # Generate x values
# x_min = 0
# x_max = 100

# # Generate x values
# x = np.linspace(x_min, x_max, 1000)

# # Instantiate reward functions using parameters that mimic the paper's figure
# designs = [
#     RewardDesign1(target=50, offset_param=250),
#     RewardDesign2(target=50, offset_param1=250, offset_param2=50),
#     RewardDesign3(target=50, offset_param=250),
#     RewardDesign4(target=50, offset_param=500),
#     RewardDesign5(target_bound_low=40, target_bound_high=60, offset_param=100),
#     RewardDesign6(target1=30, target2=70, second_peak=0.8, flat_zone=0.5,
#                   offset_param1=100, offset_param2=50, offset_param3=50, offset_param4=100)
# ]

# === Customizable boundaries for X axis ===
x_min = 0
x_max = 10000

# Generate x values
x = np.linspace(x_min, x_max, 1000)

# Instantiate reward functions using parameters that mimic the paper's figure
designs = [
    RewardDesign1(target=50, offset_param=250),
    RewardDesign2(target=50, offset_param1=250, offset_param2=50),
    RewardDesign3(target=3000, offset_param=1250000),
    RewardDesign4(target=0, offset_param=20000000),
    RewardDesign5(target_bound_low=40, target_bound_high=60, offset_param=100),
    RewardDesign6(target1=30, target2=70, second_peak=0.8, flat_zone=0.5,
                  offset_param1=100, offset_param2=50, offset_param3=50, offset_param4=100)
]

# RewardDesign3(target=e_ct_threshold, offset_param=100000) # For 1 km, 0.5 km trigger distance

# Plotting figure to match the paper style (Fig. 2)
fig, axs = plt.subplots(2, 3, figsize=(14, 6))
axs = axs.flatten()
titles = [f"(a) Design {i+1}" for i in range(6)]

for i, design in enumerate(designs):
    y = [design(float(xi)) for xi in x]  # Scalar calls
    axs[i].plot(x, y, linewidth=2)
    axs[i].set_title(titles[i], fontsize=12)

    axs[i].set_xlim(x_min, x_max)
    axs[i].set_ylim(-.1, 1.1)

    # Dynamic ticks
    axs[i].set_xticks(np.linspace(x_min, x_max, 6))  # 6 evenly spaced ticks
    axs[i].set_yticks([.0, 0.5, 1.0])

    axs[i].set_xlabel("Input value")
    axs[i].set_ylabel("Reward")

    axs[i].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

fig.suptitle("Evaluation Functions – Corresponding to Goto et al. (2023)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
