import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st
import pingouin as pg
import matplotlib.patches as mpatches

plt.rcParams["figure.figsize"] = (9, 5)
sns.set_theme(style="whitegrid")
# sns.set(font_scale=1.4)
fig_path = "/home/ltorres/projects/motion_compensation_ipf/figures/"

df = pd.read_csv("snr_cnr_measures.csv")
df["image_type"] = df["image_type"].replace(
    {
        "NoGate": "Non Gated",
        "HardGate": "Hard Gated",
        "SoftGate": "Soft Gated",
        "MotionResolved": "XD-Grasp",
        "IterativeMoCo": "iMoCo",
    }
)

color_a = "tab:blue"
color_b = "tab:orange"
color_c = "slategrey"
df_cnr = df[df.region != "Airway"]
legend_string = ["No Gating", "Hard Gating", "Soft Gating", "XD-GRASP", "iMoCo"]
# SNR Plots
ax1 = pg.plot_paired(
    data=df[df.region == "Airway"],
    dv="snr",
    within="image_type",
    subject="subject",
    dpi=150,
    figsize=(9, 5),
    order=["Non Gated", "Hard Gated", "Soft Gated", "XD-Grasp", "iMoCo"],
    boxplot=True,
    boxplot_in_front=True,
    colors=[color_a, "grey", color_b],
    boxplot_kwargs={"color": color_c, "width": 0.3},
)
ax1.set(xlabel=None, ylabel="aSNR", title="Airway aSNR")
red_patch = mpatches.Patch(color=color_b, label="Pairwise Decrease")
blue_patch = mpatches.Patch(color=color_a, label="Pairwise Increase")
plt.legend(handles=[red_patch, blue_patch], loc="upper right")

plt.savefig(fig_path + "airway_snr.png", dpi=300)
plt.close()

ax1 = pg.plot_paired(
    data=df[df.region == "Lung"],
    dv="snr",
    within="image_type",
    subject="subject",
    dpi=150,
    figsize=(9, 5),
    order=["Non Gated", "Hard Gated", "Soft Gated", "XD-Grasp", "iMoCo"],
    boxplot=True,
    boxplot_in_front=True,
    colors=[color_a, "grey", color_b],
    boxplot_kwargs={"color": color_c, "width": 0.3},
)
ax1.set(xlabel=None, ylabel="aSNR", title="Lung aSNR")
red_patch = mpatches.Patch(color=color_b, label="Pairwise Decrease")
blue_patch = mpatches.Patch(color=color_a, label="Pairwise Increase")
plt.legend(handles=[red_patch, blue_patch], loc="upper right")

plt.savefig(fig_path + "lung_snr.png", dpi=300)
plt.close()

ax1 = pg.plot_paired(
    data=df[df.region == "Muscle"],
    dv="snr",
    within="image_type",
    subject="subject",
    dpi=150,
    figsize=(9, 5),
    order=["Non Gated", "Hard Gated", "Soft Gated", "XD-Grasp", "iMoCo"],
    boxplot=True,
    boxplot_in_front=True,
    colors=[color_a, "grey", color_b],
    boxplot_kwargs={"color": color_c, "width": 0.3},
)
ax1.set(xlabel=None, ylabel="aSNR", title="Muscle aSNR")
red_patch = mpatches.Patch(color=color_b, label="Pairwise Decrease")
blue_patch = mpatches.Patch(color=color_a, label="Pairwise Increase")
plt.legend(handles=[red_patch, blue_patch], loc="upper right")

plt.savefig(fig_path + "muscle_snr.png", dpi=300)
plt.close()

ax1 = pg.plot_paired(
    data=df[df.region == "Liver"],
    dv="snr",
    within="image_type",
    subject="subject",
    dpi=150,
    figsize=(9, 5),
    order=["Non Gated", "Hard Gated", "Soft Gated", "XD-Grasp", "iMoCo"],
    boxplot=True,
    boxplot_in_front=True,
    colors=[color_a, "grey", color_b],
    boxplot_kwargs={"color": color_c, "width": 0.3},
)
ax1.set(xlabel=None, ylabel="aSNR", title="Liver aSNR")
red_patch = mpatches.Patch(color=color_b, label="Pairwise Decrease")
blue_patch = mpatches.Patch(color=color_a, label="Pairwise Increase")
plt.legend(handles=[red_patch, blue_patch], loc="upper right")

plt.savefig(fig_path + "liver_snr.png", dpi=300)
plt.close()

ax1 = pg.plot_paired(
    data=df[df.region == "Aorta"],
    dv="snr",
    within="image_type",
    subject="subject",
    dpi=150,
    figsize=(9, 5),
    order=["Non Gated", "Hard Gated", "Soft Gated", "XD-Grasp", "iMoCo"],
    boxplot=True,
    boxplot_in_front=True,
    colors=[color_a, "grey", color_b],
    boxplot_kwargs={"color": color_c, "width": 0.3},
)
ax1.set(xlabel=None, ylabel="aSNR", title="Aorta aSNR")
red_patch = mpatches.Patch(color=color_b, label="Pairwise Decrease")
blue_patch = mpatches.Patch(color=color_a, label="Pairwise Increase")
plt.legend(handles=[red_patch, blue_patch], loc="upper right")

plt.savefig(fig_path + "aorta_snr.png", dpi=300)
plt.close()

ax1 = pg.plot_paired(
    data=df[df.region == "Lung"],
    dv="cnr",
    within="image_type",
    subject="subject",
    dpi=150,
    figsize=(9, 5),
    order=["Non Gated", "Hard Gated", "Soft Gated", "XD-Grasp", "iMoCo"],
    boxplot=True,
    boxplot_in_front=True,
    colors=[color_a, "grey", color_b],
    boxplot_kwargs={"color": color_c, "width": 0.3},
)
ax1.set(xlabel=None, ylabel="CNR", title="Lung CNR")
red_patch = mpatches.Patch(color=color_b, label="Pairwise Decrease")
blue_patch = mpatches.Patch(color=color_a, label="Pairwise Increase")
plt.legend(handles=[red_patch, blue_patch], loc="upper right")

plt.savefig(fig_path + "lung_cnr.png", dpi=300)
plt.close()

ax1 = pg.plot_paired(
    data=df[df.region == "Muscle"],
    dv="cnr",
    within="image_type",
    subject="subject",
    dpi=150,
    figsize=(9, 5),
    order=["Non Gated", "Hard Gated", "Soft Gated", "XD-Grasp", "iMoCo"],
    boxplot=True,
    boxplot_in_front=True,
    colors=[color_a, "grey", color_b],
    boxplot_kwargs={"color": color_c, "width": 0.3},
)
ax1.set(xlabel=None, ylabel="CNR", title="Muscle CNR")
red_patch = mpatches.Patch(color=color_b, label="Pairwise Decrease")
blue_patch = mpatches.Patch(color=color_a, label="Pairwise Increase")
plt.legend(handles=[red_patch, blue_patch], loc="upper right")

plt.savefig(fig_path + "muscle_cnr.png", dpi=300)
plt.close()

ax1 = pg.plot_paired(
    data=df[df.region == "Liver"],
    dv="cnr",
    within="image_type",
    subject="subject",
    dpi=150,
    figsize=(9, 5),
    order=["Non Gated", "Hard Gated", "Soft Gated", "XD-Grasp", "iMoCo"],
    boxplot=True,
    boxplot_in_front=True,
    colors=[color_a, "grey", color_b],
    boxplot_kwargs={"color": color_c, "width": 0.3},
)
ax1.set(xlabel=None, ylabel="CNR", title="Liver CNR")
red_patch = mpatches.Patch(color=color_b, label="Pairwise Decrease")
blue_patch = mpatches.Patch(color=color_a, label="Pairwise Increase")
plt.legend(handles=[red_patch, blue_patch], loc="upper right")

plt.savefig(fig_path + "liver_cnr.png", dpi=300)
plt.close()

ax1 = pg.plot_paired(
    data=df[df.region == "Aorta"],
    dv="cnr",
    within="image_type",
    subject="subject",
    dpi=150,
    figsize=(9, 5),
    order=["Non Gated", "Hard Gated", "Soft Gated", "XD-Grasp", "iMoCo"],
    boxplot=True,
    boxplot_in_front=True,
    colors=[color_a, "grey", color_b],
    boxplot_kwargs={"color": color_c, "width": 0.3},
)
ax1.set(xlabel=None, ylabel="CNR", title="Aorta CNR")
red_patch = mpatches.Patch(color=color_b, label="Pairwise Decrease")
blue_patch = mpatches.Patch(color=color_a, label="Pairwise Increase")
plt.legend(handles=[red_patch, blue_patch], loc="upper right")

plt.savefig(fig_path + "aorta_cnr.png", dpi=300)
plt.close()

# print(pg.friedman(df, dv="cnr", within="image_type", subject="subject"))
print("Lungs")
print(
    pg.pairwise_ttests(
        df[df.region == "Lung"], dv="cnr", within="image_type", subject="subject", parametric=False, padjust="bonf"
    )
)
print("Muscle")
print(
    pg.pairwise_ttests(
        df[df.region == "Muscle"], dv="cnr", within="image_type", subject="subject", parametric=False, padjust="bonf"
    )
)
print("Liver")
print(
    pg.pairwise_ttests(
        df[df.region == "Liver"], dv="cnr", within="image_type", subject="subject", parametric=False, padjust="bonf"
    )
)
print("Aorta")
print(
    pg.pairwise_ttests(
        df[df.region == "Aorta"], dv="cnr", within="image_type", subject="subject", parametric=False, padjust="bonf"
    )
)
# plt.show()
# fig_path = "/home/ltorres/projects/motion_compensation_ipf/figures/"
# fig1.savefig(fig_path + "airway_snr.png", dpi=300)
# fig2.savefig(fig_path + "lung_snr.png", dpi=300)
# fig3.savefig(fig_path + "muscle_snr.png", dpi=300)
# fig4.savefig(fig_path + "liver_snr.png", dpi=300)
# fig5.savefig(fig_path + "aorta_snr.png", dpi=300)
# fig6.savefig(fig_path + "lung_cnr.png", dpi=300)
# fig7.savefig(fig_path + "muscle_cnr.png", dpi=300)
# fig8.savefig(fig_path + "liver_cnr.png", dpi=300)
# fig9.savefig(fig_path + "aorta_cnr.png", dpi=300)

# df_hard_gating = df[df.image_type == "HardGate"]
# df_imoco = df[df.image_type == "IterativeMoCo"]
# df_xdgrasp = df[df.image_type == "MotionResolved"]
# df_nogate = df[df.image_type == "NoGate"]
# df_softgate = df[df.image_type == "SoftGate"]
# df_moco = df[df.image_type == "MoCo"]

# # Hardgated vsi imoco
# t_test_results = st.wilcoxon(df_hard_gating[df_hard_gating.region == "Lung"].snr, df_imoco[df_imoco.region == "Lung"].snr)
# print(f"hard gating vs imoco lung snr {t_test_results}")
# t_test_results = st.wilcoxon(df_hard_gating[df_hard_gating.region == "Lung"].cnr, df_imoco[df_imoco.region == "Lung"].cnr)
# print(f"hard gating vs imoco lung cnr {t_test_results}")

# # Hardgated vsi motion_resolved
# t_test_results = st.wilcoxon(
#     df_hard_gating[df_hard_gating.region == "Lung"].snr, df_xdgrasp[df_xdgrasp.region == "Lung"].snr
# )
# print(f"hard gating vs xdgrasp lung snr {t_test_results}")
# t_test_results = st.wilcoxon(
#     df_hard_gating[df_hard_gating.region == "Lung"].cnr, df_xdgrasp[df_xdgrasp.region == "Lung"].cnr
# )
# print(f"hard gating vs xdgrasp lung cnr {t_test_results}")


# # Hardgated vsi imoco
