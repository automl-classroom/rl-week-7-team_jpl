import os
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from rliable import metrics
from rliable.library import get_interval_estimates
from rliable.plot_utils import plot_sample_efficiency_curve

n_seeds = 7
# Read data from different runs
# This is the toy data, you can also build a proper loop over your own runs.
df_s0_rnd = pd.read_csv(r"C:\Users\LuanL\PycharmProjects\rl-week-7-team_jpl\rl_exercises\week_7\outputs\2025-06-15\21-06-42\training_data_seed_0.csv")
df_s1_rnd = pd.read_csv(r"C:\Users\LuanL\PycharmProjects\rl-week-7-team_jpl\rl_exercises\week_7\outputs\2025-06-15\21-07-54\training_data_seed_10.csv")
df_s2_rnd = pd.read_csv(r"C:\Users\LuanL\PycharmProjects\rl-week-7-team_jpl\rl_exercises\week_7\outputs\2025-06-15\21-09-48\training_data_seed_20.csv")
df_s3_rnd = pd.read_csv(r"C:\Users\LuanL\PycharmProjects\rl-week-7-team_jpl\rl_exercises\week_7\outputs\2025-06-15\21-11-02\training_data_seed_30.csv")
df_s4_rnd = pd.read_csv(r"C:\Users\LuanL\PycharmProjects\rl-week-7-team_jpl\rl_exercises\week_7\outputs\2025-06-15\21-12-30\training_data_seed_40.csv")
df_s5_rnd = pd.read_csv(r"C:\Users\LuanL\PycharmProjects\rl-week-7-team_jpl\rl_exercises\week_7\outputs\2025-06-15\21-14-03\training_data_seed_50.csv")
df_s6_rnd = pd.read_csv(r"C:\Users\LuanL\PycharmProjects\rl-week-7-team_jpl\rl_exercises\week_7\outputs\2025-06-15\21-16-44\training_data_seed_60.csv")

df_s4_dqn = pd.read_csv(r"C:\Users\LuanL\PycharmProjects\rl-week-7-team_jpl\rl_exercises\week_4\outputs\2025-06-15\21-42-16\training_data_seed_40.csv")
df_s0_dqn = pd.read_csv(r"C:\Users\LuanL\PycharmProjects\rl-week-7-team_jpl\rl_exercises\week_4\outputs\2025-06-15\21-44-18\training_data_seed_0.csv")
df_s1_dqn = pd.read_csv(r"C:\Users\LuanL\PycharmProjects\rl-week-7-team_jpl\rl_exercises\week_4\outputs\2025-06-15\21-45-30\training_data_seed_10.csv")
df_s2_dqn = pd.read_csv(r"C:\Users\LuanL\PycharmProjects\rl-week-7-team_jpl\rl_exercises\week_4\outputs\2025-06-15\21-46-36\training_data_seed_20.csv")
df_s3_dqn = pd.read_csv(r"C:\Users\LuanL\PycharmProjects\rl-week-7-team_jpl\rl_exercises\week_4\outputs\2025-06-15\21-47-53\training_data_seed_30.csv")
df_s5_dqn = pd.read_csv(r"C:\Users\LuanL\PycharmProjects\rl-week-7-team_jpl\rl_exercises\week_4\outputs\2025-06-15\21-50-05\training_data_seed_50.csv")
df_s6_dqn = pd.read_csv(r"C:\Users\LuanL\PycharmProjects\rl-week-7-team_jpl\rl_exercises\week_4\outputs\2025-06-15\21-53-44\training_data_seed_60.csv")

df_dqn = [df_s0_dqn, df_s1_dqn, df_s2_dqn, df_s3_dqn, df_s4_dqn, df_s5_dqn, df_s6_dqn]
df_rnd = [df_s0_rnd, df_s1_rnd, df_s2_rnd, df_s3_rnd, df_s4_rnd, df_s5_rnd, df_s6_rnd]

for entry in df_dqn:
    entry["seed"] = int(entry.split("_")[1][1:])

for entry in df_rnd:
    entry["seed"] = int(entry.split("_")[1][1:])

# Add a column to distinguish between seeds
# You would do something similar for different algorithms
# df_s0["seed"] = 0

# Combine the dataframes and convert to numpy array
df_dqn = pd.concat(df_dqn, ignore_index=True)
df_rnd = pd.concat(df_rnd, ignore_index=True)

# Make sure only one set of steps is attempted to be plotted
# Obviously the steps should match in such cases!
# Steps Vereinheitlichen!!!!
steps_rnd = df_rnd["steps"].to_numpy().reshape((n_seeds, -1))[0]
steps_dqn = df_dqn["steps"].to_numpy().reshape((n_seeds, -1))[0]

# You can add other algorithms here
train_scores = {"dqn": df_dqn["rewards"].to_numpy().reshape((n_seeds, -1)),
                "rqn": df_rnd["rewards"].to_numpy().reshape((n_seeds, -1))}

# This aggregates only IQM, but other options include mean and median
# Optimality gap exists, but you obviously need optimal scores for that
# If you want to use it, check their code
iqm = lambda scores: np.array(
    [metrics.aggregate_iqm(scores[:, eval_idx]) for eval_idx in range(scores.shape[-1])]
)

iqm_scores, iqm_cis = get_interval_estimates(
    train_scores,
    iqm,
    reps=2000,
)

# This is a utility function, but you can also just use a normal line plot with the IQM and CI scores
plot_sample_efficiency_curve(
    steps + 1,
    iqm_scores,
    iqm_cis,
    algorithms=["dqn", "rnd"],
    xlabel=r"Number of Evaluations",
    ylabel="IQM Normalized Score",
)

plt.gcf().canvas.manager.set_window_title(
    "IQM Normalized Score - Sample Efficiency Curve"
)

plt.legend()
plt.tight_layout()
plt.show()
