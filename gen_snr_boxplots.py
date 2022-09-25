import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st

plt.rcParams["figure.figsize"] = (9, 5)
sns.set_theme(style="whitegrid")
# sns.set(font_scale=1.4)

df = pd.read_csv("snr_cnr_measures.csv")
# df = df[df.image_type != "MoCo"]
# df = df[df.subject != "P179_Exam1"]
# df = df[df.subject != "P153_Exam1"]
df_cnr = df[df.region != "Airway"]
legend_string = ["No Gating", "Hard Gating", "Soft Gating", "XD-GRASP", "iMoCo"]
# SNR Plots
fig1 = plt.figure()
ax1 = sns.boxplot(x="image_type", y="snr", data=df[df.region == "Airway"], orient="v")
# ax1.legend()
fig2 = plt.figure()
ax2 = sns.boxplot(x="image_type", y="snr", data=df[df.region == "Lung"], orient="v")
# ax2.legend()
fig3 = plt.figure()
ax3 = sns.boxplot(x="image_type", y="snr", data=df[df.region == "Muscle"], orient="v")
# ax3.legend()
fig4 = plt.figure()
ax4 = sns.boxplot(x="image_type", y="snr", data=df[df.region == "Liver"], orient="v")
# ax4.legend()
fig5 = plt.figure()
ax5 = sns.boxplot(x="image_type", y="snr", data=df[df.region == "Aorta"], orient="v")
# ax5.legend()
fig6 = plt.figure()
ax6 = sns.boxplot(x="image_type", y="cnr", data=df[df.region == "Lung"], orient="v")
# ax6.legend()
fig7 = plt.figure()
ax7 = sns.boxplot(x="image_type", y="cnr", data=df[df.region == "Muscle"], orient="v")
# ax7.legend()
fig8 = plt.figure()
ax8 = sns.boxplot(x="image_type", y="cnr", data=df[df.region == "Liver"], orient="v")
# ax8.legend()
fig9 = plt.figure()
ax9 = sns.boxplot(x="image_type", y="cnr", data=df[df.region == "Aorta"], orient="v")
# ax9.legend()

ax1.set(xlabel=None, ylabel="SNR", title="Airway SNR")
# ax1.set_xticklabels(ax1.get_xticklabels(), rotation=10)
ax2.set(xlabel=None, ylabel="SNR", title="Lung SNR")
ax3.set(xlabel=None, ylabel="SNR", title="Muscle SNR")
ax4.set(xlabel=None, ylabel="SNR", title="Liver SNR")
ax5.set(xlabel=None, ylabel="SNR", title="Aorta SNR")
ax6.set(xlabel=None, ylabel="CNR", title="Lung CNR")
ax7.set(xlabel=None, ylabel="CNR", title="Muscle CNR")
ax8.set(xlabel=None, ylabel="CNR", title="Liver CNR")
ax9.set(xlabel=None, ylabel="CNR", title="Aorta CNR")

# plt.show()
fig_path = "/home/ltorres/projects/motion_compensation_ipf/figures/"
fig1.savefig(fig_path + "airway_snr.png", dpi=300)
fig2.savefig(fig_path + "lung_snr.png", dpi=300)
fig3.savefig(fig_path + "muscle_snr.png", dpi=300)
fig4.savefig(fig_path + "liver_snr.png", dpi=300)
fig5.savefig(fig_path + "aorta_snr.png", dpi=300)
fig6.savefig(fig_path + "lung_cnr.png", dpi=300)
fig7.savefig(fig_path + "muscle_cnr.png", dpi=300)
fig8.savefig(fig_path + "liver_cnr.png", dpi=300)
fig9.savefig(fig_path + "aorta_cnr.png", dpi=300)

df_hard_gating = df[df.image_type == "HardGate"]
df_imoco = df[df.image_type == "IterativeMoCo"]
df_xdgrasp = df[df.image_type == "MotionResolved"]
df_nogate = df[df.image_type == "NoGate"]
df_softgate = df[df.image_type == "SoftGate"]
# df_moco = df[df.image_type == "MoCo"]

# # Hardgated vsi imoco
# t_test_results = st.ttest_ind(
#     df_hard_gating[df_hard_gating.region == "Lung"].snr, df_imoco[df_imoco.region == "Lung"].snr
# )
# print(f"hard gating vs imoco lung snr {t_test_results}")
# t_test_results = st.ttest_ind(
#     df_hard_gating[df_hard_gating.region == "Lung"].cnr, df_imoco[df_imoco.region == "Lung"].cnr
# )
# print(f"hard gating vs imoco lung cnr {t_test_results}")

# # Hardgated vsi motion_resolved
# t_test_results = st.ttest_ind(
#     df_hard_gating[df_hard_gating.region == "Lung"].snr, df_xdgrasp[df_xdgrasp.region == "Lung"].snr
# )
# print(f"hard gating vs xdgrasp lung snr {t_test_results}")
# t_test_results = st.ttest_ind(
#     df_hard_gating[df_hard_gating.region == "Lung"].cnr, df_xdgrasp[df_xdgrasp.region == "Lung"].cnr
# )
# print(f"hard gating vs xdgrasp lung cnr {t_test_results}")


# imoco vsi motion_resolved
t_test_results = st.ttest_ind(df_xdgrasp[df_xdgrasp.region == "Lung"].cnr, df_imoco[df_imoco.region == "Lung"].cnr)
print(f"imoco vs xdgrasp lung cnr {t_test_results}")

# imoco vsi  softgate
t_test_results = st.ttest_ind(df_softgate[df_softgate.region == "Lung"].cnr, df_imoco[df_imoco.region == "Lung"].cnr)
print(f"imoco vs softgate lung cnr {t_test_results}")

# imoco vsi hardgate
t_test_results = st.ttest_ind(df_hard_gating[df_hard_gating.region == "Lung"].cnr, df_imoco[df_imoco.region == "Lung"].cnr)
print(f"imoco vs hardgate lung cnr {t_test_results}")

# imoco vsi hardgate
t_test_results = st.ttest_ind(df_nogate[df_nogate.region == "Lung"].cnr, df_imoco[df_imoco.region == "Lung"].cnr)
print(f"imoco vs nogate lung cnr {t_test_results}")
