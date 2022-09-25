import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st

plt.rcParams["figure.figsize"] = (9, 5)
sns.set_theme(style="whitegrid")
fig_path = "/home/ltorres/projects/motion_compensation_ipf/figures/"

sharpness_data = pd.read_csv("sharpness_measures_no_threshold.csv")
sharpness_data = sharpness_data.melt(var_name="image_type", value_name="sharpness")

# print(sharpness_data)
# SNR Boxplots
fig1 = plt.figure()
ax1 = sns.boxplot(x="image_type", y="sharpness", data=sharpness_data, orient="v")
ax1.set(xlabel=None, ylabel="Sharpness", title="DWT Sharpness Measure")
# plt.show()
fig1.savefig(fig_path + "sharpness_boxplot.png", dpi=300)

# NoGate vs All DWT
print("---------------DWT Results No Gate vs All----------------")
t_test_results = st.ttest_ind(
    sharpness_data[sharpness_data.image_type == "No Gating"].sharpness,
    sharpness_data[sharpness_data.image_type == "Motion Resolved"].sharpness,
)
print(f"No gate vs Motion Resolved Sharpness {t_test_results}")

t_test_results = st.ttest_ind(
    sharpness_data[sharpness_data.image_type == "No Gating"].sharpness,
    sharpness_data[sharpness_data.image_type == "iMoCo"].sharpness,
)
print(f"No gate vs iMoCo Sharpness {t_test_results}")

t_test_results = st.ttest_ind(
    sharpness_data[sharpness_data.image_type == "No Gating"].sharpness,
    sharpness_data[sharpness_data.image_type == "Hard Gating"].sharpness,
)
print(f"No gate vs hardgate Sharpness {t_test_results}")

t_test_results = st.ttest_ind(
    sharpness_data[sharpness_data.image_type == "No Gating"].sharpness,
    sharpness_data[sharpness_data.image_type == "Soft Gating"].sharpness,
)
print(f"No gate vs softgate Sharpness {t_test_results}")

# IterativeMoCo vs All DWT
print("---------------DWT Results IterativeMoCo vs All----------------")
t_test_results = st.ttest_ind(
    sharpness_data[sharpness_data.image_type == "iMoCo"].sharpness,
    sharpness_data[sharpness_data.image_type == "No Gating"].sharpness,
)
print(f"iMoCo vs No Gating Sharpness {t_test_results}")

t_test_results = st.ttest_ind(
    sharpness_data[sharpness_data.image_type == "iMoCo"].sharpness,
    sharpness_data[sharpness_data.image_type == "Motion Resolved"].sharpness,
)
print(f"iMoCo vs Motion Resolved Sharpness {t_test_results}")

t_test_results = st.ttest_ind(
    sharpness_data[sharpness_data.image_type == "iMoCo"].sharpness,
    sharpness_data[sharpness_data.image_type == "Hard Gating"].sharpness,
)
print(f"iMoCo vs hardgate Sharpness {t_test_results}")

t_test_results = st.ttest_ind(
    sharpness_data[sharpness_data.image_type == "iMoCo"].sharpness,
    sharpness_data[sharpness_data.image_type == "Soft Gating"].sharpness,
)
print(f"iMoCo vs softgate Sharpness {t_test_results}")

# Motion Resolved vs All DWT
print("---------------DWT Results Motion Resolved vs All----------------")
t_test_results = st.ttest_ind(
    sharpness_data[sharpness_data.image_type == "Motion Resolved"].sharpness,
    sharpness_data[sharpness_data.image_type == "No Gating"].sharpness,
)
print(f"Motion Resolved vs No Gate Sharpness {t_test_results}")

t_test_results = st.ttest_ind(
    sharpness_data[sharpness_data.image_type == "Motion Resolved"].sharpness,
    sharpness_data[sharpness_data.image_type == "iMoCo"].sharpness,
)
print(f"Motion Resolved vs iMoCo Sharpness {t_test_results}")

t_test_results = st.ttest_ind(
    sharpness_data[sharpness_data.image_type == "Motion Resolved"].sharpness,
    sharpness_data[sharpness_data.image_type == "Hard Gating"].sharpness,
)
print(f"Motion Resolved vs hardgate Sharpness {t_test_results}")

t_test_results = st.ttest_ind(
    sharpness_data[sharpness_data.image_type == "Motion Resolved"].sharpness,
    sharpness_data[sharpness_data.image_type == "Soft Gating"].sharpness,
)
print(f"Motion Resolved vs softgate Sharpness {t_test_results}")

# t_test_results = st.ttest_ind(
#     sharpness_data[sharpness_data.image_type == "No Gating"].sharpness,
#     sharpness_data[sharpness_data.image_type == "MoCo"].sharpness,
# )
# print(f"No gate vs moco Sharpness {t_test_results}")

# Maximum Gradient Boxplots
grad_data = pd.read_csv("max_gradient.csv")
fig2 = plt.figure()
ax2 = sns.boxplot(x="image_type", y="max_gradient", data=grad_data, orient="v")
ax2.set(xlabel=None, ylabel="Diapraghm Gradient Max", title="Diaphragm Maximum Gradient")
# plt.show()
fig2.savefig(fig_path + "maximum_gradient_boxplot.png", dpi=300)

print("---------------Max Grad Results No Gate vs All----------------")
t_test_results = st.ttest_ind(
    grad_data[grad_data.image_type == "NoGate"].max_gradient,
    grad_data[grad_data.image_type == "MotionResolved"].max_gradient,
)
print(f"No gate vs Motion Resolved Max Gradient {t_test_results}")

# print(grad_data.image_type)
# nogate vs all
t_test_results = st.ttest_ind(
    grad_data[grad_data.image_type == "NoGate"].max_gradient,
    grad_data[grad_data.image_type == "IterativeMoCo"].max_gradient,
)
print(f"No gate vs iMoco  Max Gradient {t_test_results}")

t_test_results = st.ttest_ind(
    grad_data[grad_data.image_type == "NoGate"].max_gradient, grad_data[grad_data.image_type == "HardGate"].max_gradient,
)
print(f"No gate  vs HardGate Max Gradient {t_test_results}")

t_test_results = st.ttest_ind(
    grad_data[grad_data.image_type == "NoGate"].max_gradient, grad_data[grad_data.image_type == "SoftGate"].max_gradient,
)
print(f"No gate  vs SoftGate Max Gradient {t_test_results}")

print("---------------Max Grad Results IterativeMoCo vs All----------------")
# iMoCo vs all
t_test_results = st.ttest_ind(
    grad_data[grad_data.image_type == "IterativeMoCo"].max_gradient,
    grad_data[grad_data.image_type == "NoGate"].max_gradient,
)
print(f"IterativeMoCo vs NoGate Max Gradient {t_test_results}")

# print(grad_data.image_type)
t_test_results = st.ttest_ind(
    grad_data[grad_data.image_type == "IterativeMoCo"].max_gradient,
    grad_data[grad_data.image_type == "MotionResolved"].max_gradient,
)
print(f"IterativeMoCo  vs MotionResolved Max Gradient {t_test_results}")

t_test_results = st.ttest_ind(
    grad_data[grad_data.image_type == "IterativeMoCo"].max_gradient,
    grad_data[grad_data.image_type == "HardGate"].max_gradient,
)
print(f"IterativeMoCo vs HardGate Max Gradient {t_test_results}")

t_test_results = st.ttest_ind(
    grad_data[grad_data.image_type == "IterativeMoCo"].max_gradient,
    grad_data[grad_data.image_type == "SoftGate"].max_gradient,
)
print(f"IterativeMoCo  vs SoftGate Max Gradient {t_test_results}")

print("---------------Max Grad Results MotionResolved vs All----------------")
# MotionResolved vs all
t_test_results = st.ttest_ind(
    grad_data[grad_data.image_type == "MotionResolved"].max_gradient,
    grad_data[grad_data.image_type == "NoGate"].max_gradient,
)
print(f"MotionResolved vs NoGate Max Gradient {t_test_results}")

# print(grad_data.image_type)
t_test_results = st.ttest_ind(
    grad_data[grad_data.image_type == "MotionResolved"].max_gradient,
    grad_data[grad_data.image_type == "IterativeMoCo"].max_gradient,
)
print(f"MotionResolved  vs IterativeMoCo Max Gradient {t_test_results}")

t_test_results = st.ttest_ind(
    grad_data[grad_data.image_type == "MotionResolved"].max_gradient,
    grad_data[grad_data.image_type == "HardGate"].max_gradient,
)
print(f"MotionResolved vs HardGate Max Gradient {t_test_results}")

t_test_results = st.ttest_ind(
    grad_data[grad_data.image_type == "MotionResolved"].max_gradient,
    grad_data[grad_data.image_type == "SoftGate"].max_gradient,
)
print(f"MotionResolved  vs SoftGate Max Gradient {t_test_results}")
