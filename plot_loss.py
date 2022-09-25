import numpy as np
import matplotlib.pyplot as plt


loss1 = np.loadtxt(
    "/home/ltorres/data/recon/nicu/P24/diagnostics/xdgrasp_res0.75/xdgrasp_loss_lambda0.0075_res0.75.txt"
)
loss2 = np.loadtxt("/home/ltorres/data/recon/nicu/P24/diagnostics/xdgrasp_res1.0/xdgrasp_loss_lambda0.01_res1.0.txt")

plt.plot(loss1, label="Low Res")
plt.plot(loss2, label="High Res")
plt.legend()
plt.show()
