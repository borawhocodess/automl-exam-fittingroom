import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import imageio


# -------------------------------
# 1. R² Progress Plot
# -------------------------------
def plot_bo_with_tabpfn_r2(log_df: pd.DataFrame):
    plt.figure()
    plt.plot(log_df["iteration"], log_df["r2"], label="R2 per iteration")
    plt.plot(
        log_df["iteration"], log_df["best_r2"], label="Best R2 so far", linestyle="--"
    )
    plt.xlabel("Iteration")
    plt.ylabel("R2")
    plt.title("Bayesian Optimization Progress")
    plt.legend()
    plt.show()


# -------------------------------
# 2. 3D Hyperparameter Scatter Plot
# -------------------------------
def plot_bo_with_tabpfn_hyperparameter_space_3d(log_df: pd.DataFrame):
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    p = ax.scatter(
        log_df["depth"],
        log_df["l2_leaf_reg"],
        log_df["learning_rate"],
        c=log_df["r2"],
        cmap="viridis",
        s=50,
    )
    fig.colorbar(p, label="R2")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("l2_leaf_reg")
    ax.set_zlabel("learning_rate")
    ax.set_title("3D Hyperparameter Space (colored by R2)")
    plt.show()


# -------------------------------
# 3. Heatmap (Fix One Hyperparameter Slice)
# -------------------------------
def plot_bo_with_tabpfn_hyperparameter_heatmap_3d(
    log_df: pd.DataFrame, fixed_param="learning_rate"
):
    """
    Creates 2D heatmaps by fixing one hyperparameter (slice visualization).
    fixed_param: "learning_rate", "l2_leaf_reg", or "depth"
    """
    depth = log_df["depth"].values
    l2 = log_df["l2_leaf_reg"].values
    lr = log_df["learning_rate"].values
    r2 = log_df["r2"].values
    init_points = log_df["init_point"].astype(bool).sum()

    # Fix one parameter at its median value
    if fixed_param == "learning_rate":
        fixed_value = np.median(lr)
        mask = np.isclose(lr, fixed_value, rtol=0.2)  # tolerance to include near values
        x, y = depth[mask], l2[mask]
        z = r2[mask]
        x_label, y_label = "max_depth", "l2_leaf_reg"
    elif fixed_param == "l2_leaf_reg":
        fixed_value = np.median(l2)
        mask = np.isclose(l2, fixed_value, rtol=0.2)
        x, y = depth[mask], lr[mask]
        z = r2[mask]
        x_label, y_label = "max_depth", "learning_rate"
    else:  # fixed depth
        fixed_value = np.median(depth)
        mask = np.isclose(depth, fixed_value, rtol=0.2)
        x, y = l2[mask], lr[mask]
        z = r2[mask]
        x_label, y_label = "l2_leaf_reg", "learning_rate"

    # Grid
    grid_x, grid_y = np.meshgrid(
        np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
    )
    grid_r2 = griddata(points=(x, y), values=z, xi=(grid_x, grid_y), method="cubic")

    plt.figure(figsize=(8, 6))
    heatmap = plt.contourf(grid_x, grid_y, grid_r2, levels=50, cmap="viridis")
    plt.colorbar(heatmap, label="R2")

    plt.scatter(x, y, c=z, edgecolors="black", cmap="viridis", s=60)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"Hyperparameter Heatmap (Fixed {fixed_param} = {fixed_value:.4f})")
    plt.show()


# -------------------------------
# 4. GIF Creation (3 Hyperparameters)
# -------------------------------
def create_bo_with_tabpfn_gif_3d(
    log_df: pd.DataFrame, gif_name="bo_progress.gif", duration=300
):
    init_points = log_df["init_point"].astype(bool).sum()
    depth = log_df["depth"].values
    l2 = log_df["l2_leaf_reg"].values
    lr = log_df["learning_rate"].values
    r2 = log_df["r2"].values

    os.makedirs("frames", exist_ok=True)
    frame_paths = []

    for frame in range(3, len(depth) + 1):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_box_aspect([1, 1, 1])  # Equal scaling

        # Slice arrays up to current frame
        depth_f = depth[:frame]
        l2_f = l2[:frame]
        lr_f = lr[:frame]
        r2_f = r2[:frame]

        init_mask = np.arange(frame) < init_points
        bo_mask = np.arange(frame) >= init_points

        # Scatter points
        ax.scatter(
            depth_f[init_mask],
            l2_f[init_mask],
            lr_f[init_mask],
            c="gray",
            label="Init Points",
            s=60,
        )
        sc = ax.scatter(
            depth_f[bo_mask],
            l2_f[bo_mask],
            lr_f[bo_mask],
            c=r2_f[bo_mask],
            cmap="plasma",
            edgecolors="black",
            s=60,
            label="BO Points",
        )

        # Trajectory line
        if bo_mask.sum() > 1:
            ax.plot(
                depth_f[bo_mask], l2_f[bo_mask], lr_f[bo_mask], color="red", linewidth=1
            )

        ax.set_xlabel("max_depth")
        ax.set_ylabel("l2_leaf_reg")
        ax.set_zlabel("learning_rate")
        ax.set_title(f"BO Hyperparameter Search (Iteration {frame})")
        fig.colorbar(sc, label="R²")
        ax.legend()

        frame_path = f"frames/frame_{frame:03d}.png"
        plt.savefig(frame_path)
        frame_paths.append(frame_path)
        plt.close()

    with imageio.get_writer(gif_name, mode="I", duration=duration / 1000) as writer:
        for frame_path in frame_paths:
            writer.append_data(imageio.imread(frame_path))

    print(f"GIF saved as {gif_name}")


if __name__ == "__main__":
    log_df = pd.read_csv("bo_log.csv")

    plot_bo_with_tabpfn_r2(log_df)
    plot_bo_with_tabpfn_hyperparameter_space_3d(log_df)
    plot_bo_with_tabpfn_hyperparameter_heatmap_3d(log_df, fixed_param="learning_rate")
    plot_bo_with_tabpfn_hyperparameter_heatmap_3d(log_df, fixed_param="l2_leaf_reg")
    plot_bo_with_tabpfn_hyperparameter_heatmap_3d(log_df, fixed_param="depth")

    create_bo_with_tabpfn_gif_3d(log_df, gif_name="bo_progress.gif", duration=300)
