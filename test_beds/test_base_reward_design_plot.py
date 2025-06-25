import matplotlib.pyplot as plt
import numpy as np

from rl_env.reward_designs import RewardDesign1, RewardDesign2, RewardDesign3, RewardDesign4, RewardDesign5, RewardDesign6

# Generate x values
x = np.linspace(0, 100, 1000)

# Instantiate reward functions using parameters that mimic the paper's figure
designs = [
    RewardDesign1(target=50, offset_param=250),
    RewardDesign2(target=50, offset_param1=250, offset_param2=50),
    RewardDesign3(target=50, offset_param=250),
    RewardDesign4(target=50, offset_param=500),
    RewardDesign5(target_bound_low=40, target_bound_high=60, offset_param=100),
    RewardDesign6(target1=30, target2=70, second_peak=0.8, flat_zone=0.5,
                  offset_param1=100, offset_param2=50, offset_param3=50, offset_param4=100)
]

# Plotting figure to match the paper style (Fig. 2)
fig, axs = plt.subplots(2, 3, figsize=(14, 6))
axs = axs.flatten()
titles = [f"(a) Design {i+1}" for i in range(6)]

for i, design in enumerate(designs):
    y = [design(float(xi)) for xi in x]  # Scalar calls
    axs[i].plot(x, y, linewidth=2)
    axs[i].set_title(titles[i], fontsize=12)
    
    axs[i].set_xlim(0, 100)
    axs[i].set_ylim(-0.1, 1.1)
    
    axs[i].set_xticks(np.arange(0, 101, 20))
    axs[i].set_yticks([0.0, 0.5, 1.0])
    
    axs[i].set_xlabel("Input value")
    axs[i].set_ylabel("Reward")

    axs[i].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

fig.suptitle("Evaluation Functions – Corresponding to Goto et al. (2023)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
