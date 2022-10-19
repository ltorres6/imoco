import matplotlib.pyplot as plt


def plot_losses(loss_file, diagnostics_dir, name):
    loss = []

    f = open(loss_file, "r")
    for row in f:
        loss.append(float(row))

    plt.plot(loss, color="g", label="File Data")

    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Loss", fontsize=12)

    plt.title(f"{name}", fontsize=20)
    plt.legend()
    plt.savefig(diagnostics_dir + f"{name}.png")
    plt.close()


file_path = "/home/ltorres/data/new_oe_data/IterativeMoCocc_not_whitened/diagnostics/imoco_loss_refFrame0_lambda0.05_res1.0.txt"
diagnostics_path = (
    "/home/ltorres/data/new_oe_data/IterativeMoCocc_not_whitened/diagnostics/"
)
file_name = "imoco_loss_refFrame0_lambda0.05_res1.0"
plot_losses(file_path, diagnostics_path, file_name)
