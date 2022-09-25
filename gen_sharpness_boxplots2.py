import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import pingouin as pg
import matplotlib.patches as mpatches

plt.rcParams["figure.figsize"] = (9, 5)
sns.set_theme(style="whitegrid")
fig_path = "/home/ltorres/projects/motion_compensation_ipf/figures/"
# color_a = "b"
# color_b = "darkorange"
color_a = "tab:blue"
color_b = "tab:orange"
color_c = "slategrey"
sharpness_data = pd.read_csv("sharpness_measures_no_threshold.csv")
sharpness_data['image_type']=sharpness_data['image_type'].replace({'NoGate': 'Non Gated', 'HardGate': 'Hard Gated', 'SoftGate': 'Soft Gated', 'MotionResolved': 'XD-Grasp', 'IterativeMoCo': 'iMoCo'})
print()
# SNR Boxplots
ax1 = pg.plot_paired(
    data=sharpness_data,
    dv="fm",
    within="image_type",
    subject="subject",
    dpi=150,
    figsize=(9, 5),
    order=['Non Gated', "Hard Gated", "Soft Gated", "XD-Grasp", "iMoCo"],
    boxplot=True,
    boxplot_in_front=True,
    colors=[color_a, "grey", color_b],
    boxplot_kwargs={"color": color_c, "width": 0.3},
)
ax1.set(xlabel=None, ylabel="Focus Measure (Sharpness)", title="Sharpness (DWT)")
red_patch = mpatches.Patch(color=color_b, label="Pairwise Decrease")
blue_patch = mpatches.Patch(color=color_a, label="Pairwise Increase")
plt.legend(handles=[red_patch, blue_patch], loc="upper right")

plt.savefig(fig_path + "sharpness_boxplot.png", dpi=300)
plt.show()

print(pg.friedman(sharpness_data, dv="fm", within="image_type", subject="subject"))
print(pg.pairwise_ttests(sharpness_data, dv="fm", within="image_type", subject="subject", parametric=False,padjust='bonf'))


# Maximum Gradient Boxplots
grad_data = pd.read_csv("max_gradient.csv")
grad_data['image_type']=grad_data['image_type'].replace({'NoGate': 'Non Gated', 'HardGate': 'Hard Gated', 'SoftGate': 'Soft Gated', 'MotionResolved': 'XD-Grasp', 'IterativeMoCo': 'iMoCo'})
ax2 = pg.plot_paired(
    data=grad_data,
    dv="max_gradient",
    within="image_type",
    subject="subject",
    dpi=150,
    figsize=(9, 5),
    order=['Non Gated', "Hard Gated", "Soft Gated", "XD-Grasp", "iMoCo"],
    boxplot=True,
    boxplot_in_front=True,
    colors=[color_a, "grey", color_b],
    boxplot_kwargs={"color": color_c, "width": 0.3},
)
# fig2 = plt.figure()
# ax2 = sns.boxplot(x="image_type", y="max_gradient", data=grad_data, orient="v")
ax2.set(xlabel=None, ylabel="Diapraghm Gradient Max", title="Sharpness (Maximum Gradient)")
red_patch = mpatches.Patch(color=color_b, label="Pairwise Decrease")
blue_patch = mpatches.Patch(color=color_a, label="Pairwise Increase")
plt.legend(handles=[red_patch, blue_patch], loc="upper right")
plt.savefig(fig_path + "maximum_gradient_boxplot.png", dpi=300)
plt.show()
print(pg.friedman(grad_data, dv="max_gradient", within="image_type", subject="subject"))
print(pg.pairwise_ttests(grad_data, dv="max_gradient", within="image_type", subject="subject", parametric=False, padjust='bonf'))
