import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve, average_precision_score
import math

# If you have geomloss installed, you can use it for Wasserstein computation.
try:
    from geomloss import SamplesLoss
    sinkhorn_loss = SamplesLoss("sinkhorn", p=2, blur=0.05)
    HAS_GEOMLOSS = True
except ImportError:
    print("geomloss not found; skipping Wasserstein computation.")
    HAS_GEOMLOSS = False

def compute_metrics(pred, true):
    """
    Compute MSE, correlation, and R^2 between flattened predictions and true states.
    """
    diff = pred - true
    mse_val = np.mean(diff**2)

    pred_flat = pred.flatten()
    true_flat = true.flatten()
    if pred_flat.std() < 1e-12 or true_flat.std() < 1e-12:
        corr_val = 0.0
    else:
        corr_val = np.corrcoef(pred_flat, true_flat)[0,1]

    true_mean = true_flat.mean()
    ss_res = np.sum((pred_flat - true_flat)**2)
    ss_tot = np.sum((true_flat - true_mean)**2) + 1e-12
    r2_val = 1.0 - ss_res / ss_tot

    if HAS_GEOMLOSS:
        pred_t = torch.from_numpy(pred).float().to("cpu")
        true_t = torch.from_numpy(true).float().to("cpu")
        with torch.no_grad():
            wass_val = sinkhorn_loss(pred_t, true_t).item()
    else:
        wass_val = np.nan

    return mse_val, corr_val, r2_val, wass_val

def plot_predicted_vs_true_with_metrics(func_v, adata, start_bin=0, end_bin=None, n_samples=100, device="cpu", dataset_idx=None):
    """
    For each time bin in the range [start_bin, end_bin), sample cells at time bin t,
    simulate an ODE integration from t=0.0 to t=1.0 using the flow model (which is assumed to be
    trained for one time-bin transition), and compare the predicted state with the true state at t+1.
    
    Parameters:
      - func_v: your trained flow model (an instance of MLPODEF)
      - adata: an AnnData object containing observations in adata.X and time information in adata.obs["t"]
      - start_bin: starting time bin (integer)
      - end_bin: ending time bin (if None, will be set to max time in adata.obs["t"])
      - n_samples: number of cells to sample per time bin for evaluation
      - device: device on which to perform computation
      - dataset_idx: if your model uses knockout masks (i.e. trained on multiple datasets), pass the index
      
    This function plots a scatter plot for each transition (using PCA if d > 2) and a bar chart of metrics.
    """
    # Ensure the model is on the correct device.
    func_v.to(device)

    # Determine the range of time bins
    times_all = np.unique(adata.obs["t"].values)
    times_all = np.sort(times_all.astype(np.int32))
    if end_bin is None:
        end_bin = times_all.max()

    overall_metrics = []  # to store metrics for each bin

    for tb in range(start_bin, end_bin):
        # Get cells at time bin tb and tb+1
        cells_t0 = torch.from_numpy(adata.X[adata.obs["t"] == tb]).float()
        cells_t1 = torch.from_numpy(adata.X[adata.obs["t"] == tb + 1]).float()

        n0 = cells_t0.size(0)
        n1 = cells_t1.size(0)
        if n0 == 0 or n1 == 0:
            print(f"Skipping time bin {tb}->{tb+1} due to insufficient data.")
            continue

        n_samples_bin = min(n_samples, n0, n1)
        idx0 = np.random.choice(n0, size=n_samples_bin, replace=False)
        idx1 = np.random.choice(n1, size=n_samples_bin, replace=False)
        x0 = cells_t0[idx0].to(device)
        x1 = cells_t1[idx1].to(device)

        # Prepare initial conditions for ODE integration:
        # We assume each transition is over a unit time interval.
        times = torch.tensor([0.0, 1.0], device=device)
        # The model expects input with shape [n_samples, 1, d]
        x0_reshaped = x0.unsqueeze(1)

        # Define a wrapper function that “fixes” the time bin value in the flow model.
        # Since our model's forward signature is (t, x, dataset_idx), we fix t to be tb.
        def flow_func(t, x):
            # Here, we ignore the integration time t (which spans 0->1) and use the trained time bin tb.
            t_fixed = torch.full((x.shape[0],), float(tb), device=device)
            if dataset_idx is not None:
                return func_v(t_fixed, x, dataset_idx)
            else:
                return func_v(t_fixed, x)

        # Simulate the prediction via ODE integration:
        with torch.no_grad():
            z_pred = odeint(flow_func, x0_reshaped, times, method="dopri5")
        # Get the predicted state at time=1 (shape: [n_samples, 1, d] -> [n_samples, d])
        z_pred_final = z_pred[-1].squeeze(1).cpu().numpy()
        true_state = x1.cpu().numpy()

        # If the number of variables (d) is greater than 2, use PCA for visualization.
        d = x0.shape[1]
        plt.figure(figsize=(6,6))
        if d > 2:
            pca = PCA(n_components=2)
            combined = np.vstack([z_pred_final, true_state])
            combined_2d = pca.fit_transform(combined)
            pred_2d = combined_2d[:n_samples_bin]
            true_2d = combined_2d[n_samples_bin:]
            plt.scatter(pred_2d[:,0], pred_2d[:,1], alpha=0.5, label="Predicted")
            plt.scatter(true_2d[:,0], true_2d[:,1], alpha=0.5, label="True")
            plt.title(f"Time bin {tb}->{tb+1}: PCA (2D) comparison")
        else:
            plt.scatter(z_pred_final[:,0], z_pred_final[:,1], alpha=0.5, label="Predicted")
            plt.scatter(true_state[:,0], true_state[:,1], alpha=0.5, label="True")
            plt.title(f"Time bin {tb}->{tb+1}: Direct 2D scatter")
        plt.legend()
        plt.show()

        # Compute metrics: MSE, correlation, R^2, and Wasserstein distance if available.
        mse_val, corr_val, r2_val, wass_val = compute_metrics(z_pred_final, true_state)
        overall_metrics.append((tb, mse_val, corr_val, r2_val, wass_val))
        print(f"Time bin {tb}->{tb+1}: MSE={mse_val:.4f}, Corr={corr_val:.4f}, R^2={r2_val:.4f}, Wass={wass_val:.4f}")

        # Plot a bar chart of metrics for this time bin.
        metrics = [mse_val, corr_val, r2_val, wass_val]
        labels  = ["MSE", "Corr", "R^2", "Wass"]
        plt.figure(figsize=(5,4))
        plt.bar(range(len(metrics)), metrics, color=["skyblue","orange","green","red"])
        plt.xticks(range(len(metrics)), labels)
        plt.title(f"Metrics for time bin {tb}->{tb+1}")
        plt.ylim(bottom=min(0, min(metrics)-0.1))
        plt.show()

        print(f"Time bin {tb}->{tb+1}: MSE={mse_val:.4f}, Corr={corr_val:.4f}, R^2={r2_val:.4f}, Wass={wass_val:.4f}")
    return mse_val, corr_val, r2_val, wass_val

def plot_all_datasets_trajectories(func, adatas_list, time_bin=0, n_samples=100, device="cpu", dataset_idxs=None):
    """
    Plot predicted vs. true trajectories for each dataset in a grid.
    """
    n_datasets = len(adatas_list)
    cols = 4  # adjust as needed
    rows = int(np.ceil(n_datasets / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    axes = axes.flatten()
    
    for i, adata in enumerate(adatas_list):
        ds_idx = dataset_idxs[i] if dataset_idxs is not None else None
        cells_t0 = torch.from_numpy(adata.X[adata.obs["t"] == time_bin]).float()
        cells_t1 = torch.from_numpy(adata.X[adata.obs["t"] == time_bin + 1]).float()
        n0 = cells_t0.size(0)
        n1 = cells_t1.size(0)
        if n0 == 0 or n1 == 0:
            print(f"Dataset {i}: Insufficient data for time bin {time_bin}->{time_bin+1}")
            continue
        n_samples_ds = min(n_samples, n0, n1)
        idx0 = np.random.choice(n0, n_samples_ds, replace=False)
        idx1 = np.random.choice(n1, n_samples_ds, replace=False)
        x0 = cells_t0[idx0].to(device)
        x1 = cells_t1[idx1].to(device)
        
        # Prepare initial condition for ODE integration
        times = torch.tensor([0.0, 1.0], device=device)
        x0_reshaped = x0.unsqueeze(1)
        
        def flow_func(t, x):
            t_fixed = torch.full((x.shape[0],), float(time_bin), device=device)
            if ds_idx is not None:
                return func(t_fixed, x, ds_idx)
            else:
                return func(t_fixed, x)
        
        with torch.no_grad():
            z_pred = odeint(flow_func, x0_reshaped, times, method="dopri5")
        z_pred_final = z_pred[-1].squeeze(1).cpu().numpy()
        true_state = x1.cpu().numpy()
        
        ax = axes[i]
        d = x0.shape[1]
        if d > 2:
            pca = PCA(n_components=2)
            combined = np.vstack([z_pred_final, true_state])
            combined_2d = pca.fit_transform(combined)
            pred_2d = combined_2d[:n_samples_ds]
            true_2d = combined_2d[n_samples_ds:]
            ax.scatter(pred_2d[:,0], pred_2d[:,1], alpha=0.5, label="Predicted")
            ax.scatter(true_2d[:,0], true_2d[:,1], alpha=0.5, label="True")
            ax.set_title(f"Dataset {i}: {time_bin}->{time_bin+1}")
        else:
            ax.scatter(z_pred_final[:,0], z_pred_final[:,1], alpha=0.5, label="Predicted")
            ax.scatter(true_state[:,0], true_state[:,1], alpha=0.5, label="True")
            ax.set_title(f"Dataset {i}: {time_bin}->{time_bin+1}")
        ax.legend()
    
    # Remove any extra subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()
