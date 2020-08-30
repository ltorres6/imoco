import numpy as np
import matplotlib.pyplot as plt
def normalize(arr):
    return arr/np.max(arr)

loss001 = normalize(np.load('/export/home/ltorres/dataMrcv/FainLab/recon/ipf/103-005/mri/20160114/PreContrastMotionResolved/diagnostics/loss_frm-1_lambda0.001_res1.0.npy'))
loss01 =  normalize(np.load('/export/home/ltorres/dataMrcv/FainLab/recon/ipf/103-005/mri/20160114/PreContrastMotionResolved/diagnostics/loss_frm-1_lambda0.01_res1.0.npy'))
loss02 =  normalize(np.load('/export/home/ltorres/dataMrcv/FainLab/recon/ipf/103-005/mri/20160114/PreContrastMotionResolved/diagnostics/loss_frm-1_lambda0.02_res1.0.npy'))
loss03 =  normalize(np.load('/export/home/ltorres/dataMrcv/FainLab/recon/ipf/103-005/mri/20160114/PreContrastMotionResolved/diagnostics/loss_frm-1_lambda0.03_res1.0.npy'))
loss04 =  normalize(np.load('/export/home/ltorres/dataMrcv/FainLab/recon/ipf/103-005/mri/20160114/PreContrastMotionResolved/diagnostics/loss_frm-1_lambda0.04_res1.0.npy'))
loss05 =  normalize(np.load('/export/home/ltorres/dataMrcv/FainLab/recon/ipf/103-005/mri/20160114/PreContrastMotionResolved/diagnostics/loss_frm-1_lambda0.05_res1.0.npy'))
loss06 =  normalize(np.load('/export/home/ltorres/dataMrcv/FainLab/recon/ipf/103-005/mri/20160114/PreContrastMotionResolved/diagnostics/loss_frm-1_lambda0.06_res1.0.npy'))
loss07 =  normalize(np.load('/export/home/ltorres/dataMrcv/FainLab/recon/ipf/103-005/mri/20160114/PreContrastMotionResolved/diagnostics/loss_frm-1_lambda0.07_res1.0.npy'))
loss08 =  normalize(np.load('/export/home/ltorres/dataMrcv/FainLab/recon/ipf/103-005/mri/20160114/PreContrastMotionResolved/diagnostics/loss_frm-1_lambda0.08_res1.0.npy'))
loss09 =  normalize(np.load('/export/home/ltorres/dataMrcv/FainLab/recon/ipf/103-005/mri/20160114/PreContrastMotionResolved/diagnostics/loss_frm-1_lambda0.09_res1.0.npy'))
loss1 =  normalize(np.load('/export/home/ltorres/dataMrcv/FainLab/recon/ipf/103-005/mri/20160114/PreContrastMotionResolved/diagnostics/loss_frm-1_lambda0.1_res1.0.npy'))

x = np.arange(0,20)
print(x.shape)
print(loss001.shape)
plt.plot(x,loss001,x,loss01,x,loss02,x,loss03,x,loss04,x,loss05,x,loss06,x,loss07,x,loss08,x,loss09,x,loss1)
plt.show()