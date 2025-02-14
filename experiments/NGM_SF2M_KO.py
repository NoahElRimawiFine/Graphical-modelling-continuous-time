import numpy as np
import torch
import sys
sys.path.append("../../")
import matplotlib.pyplot as plt
import NMC as models
import importlib
import os
import ot
import glob
import pandas as pd
import scipy.sparse as sp
import anndata as ad
from tqdm import tqdm
from torchdiffeq import odeint
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import copy
from sklearn.metrics import precision_recall_curve, average_precision_score
import torchsde
from src import util
from sf2m_utils import SDE, torch_wrapper, wasserstein
from plot_utils import *
import fm

T = 5
class DataLoader:
    def __init__(self, data_path, dataset_type="Synthetic"):
        """
        Initialize DataLoader

        Args:
            data_path: Path to data directory
            dataset_type: Either "Synthetic" or "Curated"
        """
        self.data_path = os.path.join(data_path, dataset_type)
        self.dataset_type = dataset_type
        self.adatas = None
        self.kos = None
        self.true_matrix = None

    def load_data(self):
        """Load and preprocess data"""
        if self.dataset_type == "Synthetic":
            paths = glob.glob(
                os.path.join(self.data_path, "dyn-TF/dyn-TF*-1")
            ) + glob.glob(os.path.join(self.data_path, "dyn-TF_ko*/dyn-TF*-1"))
        elif self.dataset_type == "Curated":
            paths = glob.glob(os.path.join(self.data_path, f"HSC*/HSC*-1"))
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

        self.adatas = [util.load_adata(p) for p in paths]

        df = pd.read_csv(os.path.join(os.path.dirname(paths[0]), "refNetwork.csv"))

        n_genes = self.adatas[0].n_vars

        self.true_matrix = pd.DataFrame(
            np.zeros((n_genes, n_genes), int),
            index=self.adatas[0].var.index,
            columns=self.adatas[0].var.index,
        )

        for i in range(df.shape[0]):
            _i = df.iloc[i, 1]
            _j = df.iloc[i, 0]  
            _v = {"+": 1, "-": -1}[df.iloc[i, 2]]  # interaction type
            self.true_matrix.loc[_i, _j] = _v

        t_bins = np.linspace(0, 1, T + 1)[:-1]
        for adata in self.adatas:
            adata.obs["t"] = np.digitize(adata.obs.t_sim, t_bins) - 1

        self.kos = []
        for p in paths:
            try:
                self.kos.append(os.path.basename(p).split("_ko_")[1].split("-")[0])
            except:
                self.kos.append(None)

        self.gene_to_index = {
            gene: idx for idx, gene in enumerate(self.adatas[0].var.index)
        }
        self.ko_indices = []
        for ko in self.kos:
            if ko is None:
                self.ko_indices.append(None)
            else:
                self.ko_indices.append(self.gene_to_index[ko])

def sample_map(pi, batch_size, replace=True):
    """
    Randomly pick (i, j) from the coupling matrix pi (shape [n0, n1]).
    Returns arrays of row indices i and column indices j.
    """
    p = pi.flatten()
    p = p / p.sum()
    choices = np.random.choice(len(p), size=batch_size, replace=replace, p=p)
    i = choices // pi.shape[1]
    j = choices % pi.shape[1]
    return i, j


def sample_plan(x0, x1, pi, batch_size, device="cpu"):
    """
    Given x0 in [n0, d], x1 in [n1, d], and pi in [n0, n1],
    sample a batch of (x0, x1) pairs according to pi.
    """
    i, j = sample_map(pi, batch_size)
    return torch.tensor(x0[i], dtype=torch.float32, device=device), torch.tensor(
        x1[j], dtype=torch.float32, device=device
    )


def brownian_bridge(x0, x1, tau, sigma=0.1):
    """
    Construct a Brownian bridge from x0->x1 at fraction tau in [0,1].
    x0, x1: shape [batch_size, d]
    tau: shape [batch_size, 1]
    sigma: noise scale
    """
    mean_ = (1 - tau) * x0 + tau * x1
    var_ = (sigma**2) * tau * (1 - tau)
    # sample x(tau) = mean + sqrt(var)*epsilon
    eps = torch.randn_like(x0)
    x_tau = mean_ + torch.sqrt(var_.clamp_min(1e-10)) * eps

    # bridging score: s = -(x - mean)/var
    s_true = -(x_tau - mean_) / var_.clamp_min(1e-10)

    denom = 2 * tau * (1 - tau) + 1e-10
    u = ((1 - 2 * tau) / denom) * (x_tau - mean_) + (x1 - x0)
    return x_tau, s_true, u


def prepare_time_binned_data(adata, time_column="t"):
    """
    Groups cells by their time bins and returns a list of tensors.

    Args:
        adata (AnnData): The AnnData object containing cell data.
        time_column (str): The column in adata.obs indicating time bins.

    Returns:
        List[torch.Tensor]: A list where each element is a tensor of cells at a specific time bin.
    """
    num_time_bins = adata.obs[time_column].nunique()
    time_bins = sorted(adata.obs[time_column].unique())
    grouped_data = []
    for t in time_bins:
        cells_t = adata[adata.obs[time_column] == t].X
        if isinstance(cells_t, sp.spmatrix):
            cells_t = cells_t.toarray()
        grouped_data.append(torch.from_numpy(cells_t).float())
    return grouped_data


def normalize_data(grouped_data):
    """
    Applies Z-score normalization to each gene across all cells.

    Args:
        grouped_data (List[torch.Tensor]): List of tensors grouped by time bins.

    Returns:
        List[torch.Tensor]: Normalized data.
    """
    all_cells = torch.cat(grouped_data, dim=0)
    scaler = StandardScaler()
    all_cells_np = all_cells.numpy()
    scaler.fit(all_cells_np)

    normalized_data = []
    for tensor in grouped_data:
        normalized = torch.from_numpy(scaler.transform(tensor.numpy())).float()
        normalized_data.append(normalized)
    return normalized_data, scaler

def build_knockout_mask(d, ko_idx):
    """
    Build a [d, d] adjacency mask for a knockout of gene ko_idx.
    If ko_idx is None, return a mask of all ones (wild-type).
    """
    if ko_idx is None:
        # No knockout => no edges removed
        return np.ones((d, d), dtype=np.float32)
    else:
        mask = np.ones((d, d), dtype=np.float32)
        g = ko_idx
        # Zero row g => remove outgoing edges from gene g
        #mask[g, :] = 0.0
        # Zero column g => remove incoming edges to gene g
        mask[:, g] = 0.0
        mask[g, g] = 1.0
        return mask
    
def build_entropic_otfms(adatas, T, sigma, dt):
    """
    Returns a list of EntropicOTFM objects, one per dataset.
    """
    otfms = []
    for adata in adatas:
        x_tensor = torch.tensor(adata.X, dtype=torch.float32)
        t_idx = torch.tensor(adata.obs["t"], dtype=torch.long)
        model = fm.EntropicOTFM(
            x=x_tensor,
            t_idx=t_idx,
            dt=dt,
            sigma=sigma,
            T=T,
            dim=x_tensor.shape[1],
            device=torch.device("cpu")
        )
        otfms.append(model)
    return otfms


def compute_pi_entropic_fixed(x0, x1, reg=1e-2, numItermax=10000, ko_index=None, cost=1e9):
    """
    Computes an entropic OT plan between x0 and x1 using the Sinkhorn algorithm.
    """
    x0_np = x0.cpu().numpy()
    x1_np = x1.cpu().numpy()
    a = ot.unif(x0_np.shape[0])  # uniform distribution over rows
    b = ot.unif(x1_np.shape[0])  # uniform distribution over columns
    # Cost matrix: squared Euclidean distance
    M = np.sum((x0_np[:, None, :] - x1_np[None, :, :])**2, axis=2)
    # if ko_index is not None:
    #     ko0 = (x0_np[:, ko_index] < 1)
    #     ko1 = (x1_np[:, ko_index] < 1)
    #     mismatch = (ko0[:,None] != ko1[None, :])
    #     M[mismatch] = cost
    pi = ot.sinkhorn(a, b, M, reg=reg, numItermax=numItermax)
    return pi

def compute_all_pis_fixed(adata, t, reg=1e-2, ko_index=None):
    """
    Precompute entropic OT for each time bin using a fixed plan (single dataset).
    
    Returns:
        all_pis: list of length t, where all_pis[time_bin] = pi_matrix (or None)
    """
    all_pis = []
    for time_bin in range(t):
        # Extract cells belonging to time_bin and time_bin+1
        x0 = adata.X[adata.obs["t"] == time_bin]
        x1 = adata.X[adata.obs["t"] == time_bin + 1]
        
        # Convert to torch tensors
        x0 = torch.from_numpy(x0).float()
        x1 = torch.from_numpy(x1).float()
        
        if x0.size(0) == 0 or x1.size(0) == 0:
            pi = None
        else:
            pi = compute_pi_entropic_fixed(x0, x1, reg=reg, ko_index=ko_index)
        
        all_pis.append(pi)
    return all_pis


class MLP(nn.Module):
    def __init__(
        self,
        d=2,
        hidden_sizes=[100,],
        activation=nn.ReLU,
        time_varying=True,
        conditional=False,
        conditional_dim=0  # dimension of the knockout or condition
    ):
        super(MLP, self).__init__()
        self.time_varying = time_varying
        self.conditional = conditional

        input_dim = d
        if self.time_varying:
            input_dim += 1
        if self.conditional:
            input_dim += conditional_dim

        hidden_sizes = copy.copy(hidden_sizes)
        hidden_sizes.insert(0, input_dim)  # first layer's input size
        hidden_sizes.append(d)             # final layer is dimension d

        layers = []
        for i in range(len(hidden_sizes) - 1):
            in_size = hidden_sizes[i]
            out_size = hidden_sizes[i+1]
            layers.append(nn.Linear(in_size, out_size))
            # activation except for the last layer
            if i < len(hidden_sizes) - 2:
                layers.append(activation())

        self.net = nn.Sequential(*layers)

        # Weight init
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.normal_(m.bias, mean=0, std=0)

    def forward(self, t, x, cond=None):
        inputs = [x]
        if self.time_varying:
            if t.dim() == 1:
                t = t.unsqueeze(-1)
            inputs.append(t)

        if self.conditional:
            if cond is None:
                raise ValueError("Conditional flag = True, but no 'cond' input provided.")
            Bx = x.shape[0]
            if cond.dim() == 1:
                cond = cond.unsqueeze(0).expand(Bx, -1)
            elif cond.shape[0] != Bx:
                raise ValueError(
                    f"cond batch size ({cond.shape[0]}) != x batch size ({Bx}). "
                )
            inputs.append(cond)

        # cat along dim=1 => shape [batch_size, (d + time + cond_dim)]
        net_in = torch.cat(inputs, dim=1)
        return self.net(net_in)


def train_with_fmot_scorematching(
    func_v,
    func_s,
    adatas_list,
    otfms,
    cond_matrix,
    alpha=0.5,
    reg=1e-5,
    n_steps=2000,
    batch_size=64,
    device="cpu",
    lr=1e-3,
    true_mat = None
):
    """
    Combine flow matching + score matching with multiple datasets
    """
    func_v.to(device)
    func_s.to(device)
    optim = torch.optim.AdamW(
        list(func_v.parameters()) + list(func_s.parameters()), lr=lr
    )

    loss_history = []

    save_dir = "training_visuals"
    os.makedirs(save_dir, exist_ok=True)

    def proximal(w, dims, lam=0.1, eta=0.1):
        with torch.no_grad():
            d = dims[0]
            d_hidden = dims[1]
            wadj = w.view(d, d_hidden, d)
            tmp = torch.sum(wadj**2, dim=1).sqrt() - lam * eta
            alpha_ = torch.clamp(tmp, min=0)
            v_ = F.normalize(wadj, dim=1) * alpha_[:, None, :]
            w.copy_(v_.view(-1, d))

    for i in tqdm(range(n_steps)):
        ds_idx = np.random.randint(0, len(adatas_list))
        model = otfms[ds_idx]
        cond_vector = cond_matrix[ds_idx]

        _x, _s, _u, _t, _t_orig = model.sample_bridges_flows(batch_size=batch_size)
        optim.zero_grad()

        # Reshape inputs for MLPODEF
        s_input = _x.unsqueeze(1)
        v_input = _x.unsqueeze(1)
        t_input = _t.unsqueeze(1)

        B = _x.shape[0]
        cond_expanded = cond_vector.repeat(B // 164 + 1, 1)[:B]

        # Get model outputs and reshape
        s_fit = func_s(_t, _x, cond_expanded).squeeze(1)
        # v_fit = v(t_input, v_input).squeeze(1)
        v_fit = func_v(t_input, v_input, ds_idx).squeeze(1) - model.sigma**2/2 * func_s(_t, _x, cond_expanded)

        L_score = torch.mean((_t_orig * (1 - _t_orig)) * (s_fit - _s) ** 2)
        L_flow = torch.mean((v_fit * model.dt - _u) ** 2)

        L_reg = func_v.l2_reg() + func_v.fc1_reg()
        if i < 100: # train score for first few iters 
            L = alpha * L_score
        else:
            L = alpha * L_score + (1 - alpha) * L_flow + reg * L_reg

        with torch.no_grad():
            if i % 100 == 0:
                print(L_score.item(), L_flow.item(), L_reg.item())
            loss_history.append(L.item())

        L.backward()
        optim.step()

        # proximal(s.fc1.weight, s.dims, lam=s.GL_reg, eta=0.01)
        proximal(func_v.fc1.weight, func_v.dims, lam=func_v.GL_reg, eta=0.01)

        if i % 1000 == 0:
            with torch.no_grad():
                graph_sm = func_v.causal_graph()

                # Save Learned Graph
                plt.figure(figsize=(8, 6))
                plt.matshow(graph_sm, cmap="Reds", fignum=False)
                plt.gca().invert_yaxis()
                plt.colorbar()
                plt.title("Learned graph")
                graph_path = os.path.join(save_dir, f"learned_graph_step_{i}.png")
                plt.savefig(graph_path, dpi=300, bbox_inches="tight")
                plt.close()

                # Save Precision-Recall Curve
                plt.figure(figsize=(8, 6))
                y_true = np.abs(np.sign(true_mat).astype(int).flatten())
                y_pred = np.abs(graph_sm.flatten())
                prec, rec, thresh = precision_recall_curve(y_true, y_pred)
                avg_prec = average_precision_score(y_true, y_pred)
                plt.plot(rec, prec, label=f"Flow-based (AP = {avg_prec:.2f})")
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title(
                    f"Precision-Recall Curve\nAUPR ratio = {avg_prec/np.mean(np.abs(true_mat) > 0):.2f}"
                )
                plt.legend()
                plt.grid(True)
                pr_curve_path = os.path.join(save_dir, f"precision_recall_step_{i}.png")
                plt.savefig(pr_curve_path, dpi=300, bbox_inches="tight")
                plt.close()

            print(
                f"Step={i}, dataset={ds_idx}, L_score={L_score.item():.4f}, "
                f"L_flow={L_flow.item():.4f}, L_reg={L_reg:.4f}, "
                f"Saved figures to {save_dir}"
            )

    plt.plot(loss_history)
    plt.title("Score+Flow Matching Loss")
    plt.xlabel("training step")
    plt.ylabel("loss")
    plt.show()

    return loss_history, func_v, func_s


def simulate_trajectory(flow_model, score_model, x0, t_span, num_steps, sigma=0.1, device="cpu", use_sde=False):
    """
    Simulate trajectory using either ODE or SDE integration over a specified time interval.

    Args:
        flow_model: The trained flow model.
        score_model: The trained score model.
        x0: Initial conditions tensor [batch_size, n_features].
        t_span: Tuple (t0, t_end) specifying the simulation time interval.
        num_steps: Number of time points to simulate (including endpoints).
        sigma: Noise scale for SDE.
        device: Device to run simulation on.
        use_sde: If True, use SDE integration; if False, use ODE integration.

    Returns:
        trajectory: Simulated trajectory tensor [num_steps, batch_size, n_features].
    """
    x0 = x0.to(device)
    t0, t_end = t_span
    ts = torch.linspace(t0, t_end, num_steps, device=device)

    # Define a unified drift function.
    def drift(t, x):
        # Create a time tensor for each sample.
        t_batch = torch.full((x.shape[0],), t.item(), device=x.device)
        flow = flow_model(t_batch, x.unsqueeze(1)).squeeze(1)
        score = score_model(t_batch, x, )
        # Use the drift formulation: flow - (sigma^2 / 2) * score.
        return flow - (sigma**2 / 2) * score

    if use_sde:
        # For SDE integration, we also need a diffusion function.
        def diffusion(t, x):
            return sigma * torch.ones_like(x)
        
        # Wrap the drift and diffusion functions in an SDE module that is compatible with torchsde.
        class FlowSDE(torch.nn.Module):
            # Specify attributes required by torchsde.
            noise_type = "diagonal"
            sde_type = "ito"
            def __init__(self, drift_fn, diffusion_fn):
                super().__init__()
                self.drift_fn = drift_fn
                self.diffusion_fn = diffusion_fn
            def f(self, t, x):
                return self.drift_fn(t, x)
            def g(self, t, x):
                return self.diffusion_fn(t, x)
        
        sde = FlowSDE(drift, diffusion)
        with torch.no_grad():
            # Use an Euler scheme with a small dt.
            trajectory = torchsde.sdeint(sde, x0, ts, method="euler", dt=1e-2)
    else:
        # Use the ODE integrator.
        with torch.no_grad():
            trajectory = odeint(drift, x0, ts, method="dopri5")
    
    return trajectory.cpu()


def train_and_evaluate_with_holdout(adatas,
                                    held_out_time,
                                    num_variables,
                                    kos, 
                                    ko_indices,
                                    true_matrix,
                                    hidden_dim=200,
                                    n_steps=5000,
                                    device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Train the model on data excluding one timepoint and then evaluate trajectory simulation
    on the held-out time.

    We simulate trajectories over a relative time interval.
    
    Args:
        adatas: List of AnnData objects.
        held_out_time: The timepoint to hold out.
        num_variables: Number of variables (e.g., genes).
        hidden_dim: Hidden layer dimension.
        n_steps: Number of training steps.
        device: Computation device.
    
    Returns:
        avg_distances: Dictionary with average Wasserstein distances for ODE and SDE simulations.
        flow_model: Trained flow model.
        score_model: Trained score model.
    """
    batch_size = 164

    # want to create a [8, n, 8] matrix that is one hot encoded and will be selected depending on dataset idx
    conditionals = []
    for i, ad in enumerate(kos):
        cond_matrix = torch.zeros(batch_size, 8)
        if ad is not None:
            cond_matrix[:,i] = 1
        conditionals.append(cond_matrix)
   
    knockout_masks = []
    for i, ad in enumerate(adatas):
        d = ad.X.shape[1]
        mask_i = build_knockout_mask(d, ko_indices[i])  # returns [d,d]
        knockout_masks.append(mask_i)

    wt_idx = [i for i, ko in enumerate(kos) if ko is None]
    ko_idx = [i for i, ko in enumerate(kos) if ko is not None]
    adatas_wt = [adatas[i] for i in wt_idx]
    adatas_ko = [adatas[i] for i in ko_idx]
    num_variables = 8
    hidden_dim = 200
    dims = [num_variables, hidden_dim, 1]
    t = adatas[0].obs["t"].max()

    func_v = models.MLPODEF(dims=dims, GL_reg=0.01, bias=True, knockout_masks=knockout_masks)
    score_net = MLP(d=num_variables, hidden_sizes=[hidden_dim], time_varying=True, conditional=True, conditional_dim=8)

    # Compute pis for all datasets
    all_pis_list = []
    for i, ad in enumerate(zip(adatas, ko_indices)):
        T_local = ad[0].obs["t"].max()
        pi_list = compute_all_pis_fixed(ad[0], T_local, reg=1e-1, ko_index=ad[1])
        all_pis_list.append(pi_list)

    loss_history, flow_model, score_model = train_with_fmot_scorematching(
        func_v=func_v,
        func_s=score_net,
        adatas_list=adatas,  
        all_pis_list=all_pis_list,
        t=t,
        cond_matrix=conditionals,
        sigma=1.0,
        dt=1.0,
        alpha=0.1,
        reg=1e-6,
        n_steps=180000,
        batch_size=batch_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
        lr=3e-3,
        true_mat = true_matrix
    )

    # Evaluate on held-out timepoint.
    distances = []
    for adata in adatas:
        x0 = torch.from_numpy(adata.X[adata.obs["t"] == held_out_time - 1]).float()
        true_dist = torch.from_numpy(adata.X[adata.obs["t"] == held_out_time]).float()

        if len(x0) == 0 or len(true_dist) == 0:
            continue

        traj_ode = simulate_trajectory(flow_model, score_model, x0, t_span=(0.0, 1.0), num_steps=2, use_sde=False)
        traj_sde = simulate_trajectory(flow_model, score_model, x0, t_span=(0.0, 1.0), num_steps=2, use_sde=True)

        w_dist_ode = wasserstein(traj_ode[-1], true_dist)
        w_dist_sde = wasserstein(traj_sde[-1], true_dist)

        distances.append({"ode": w_dist_ode, "sde": w_dist_sde})

    avg_distances = {
        "ode": np.mean([d["ode"] for d in distances]),
        "sde": np.mean([d["sde"] for d in distances]),
    }

    return avg_distances, flow_model, score_model


def main():
    T = 5
    data_loader = DataLoader("../../data/simulation", dataset_type="Synthetic")
    data_loader.load_data()
    adatas, kos, ko_indices, true_matrix = (
        data_loader.adatas,
        data_loader.kos,
        data_loader.ko_indices,
        data_loader.true_matrix.values,
    )
    batch_size = 164
    n = adatas[0].X.shape[1]

    # want to create a [8, n, 8] matrix that is one hot encoded and will be selected depending on dataset idx
    conditionals = []
    for i, ad in enumerate(kos):
        cond_matrix = torch.zeros(batch_size, n)
        if ad is not None:
            cond_matrix[:,i] = 1
        conditionals.append(cond_matrix)
   
    knockout_masks = []
    for i, ad in enumerate(adatas):
        d = ad.X.shape[1]
        mask_i = build_knockout_mask(d, ko_indices[i])  # returns [d,d]
        knockout_masks.append(mask_i)

    wt_idx = [i for i, ko in enumerate(kos) if ko is None]
    ko_idx = [i for i, ko in enumerate(kos) if ko is not None]
    adatas_wt = [adatas[i] for i in wt_idx]
    adatas_ko = [adatas[i] for i in ko_idx]
    dims = [n, 100, 1]
    t = adatas[0].obs["t"].max()

    #func_v = models.MLPODEF(dims=dims, GL_reg=0.01, bias=True)
    func_v = models.MLPODEF1(dims=dims, GL_reg=0.04, bias=True, knockout_masks=knockout_masks)
    score_net = MLP(d=n, hidden_sizes=[100,100], time_varying=True, conditional=True, conditional_dim=n)
    # score_net = fm.MLP(d=n, hidden_sizes = [64, 64], time_varying=True)

    otfms = build_entropic_otfms(adatas, T, sigma= 1.0, dt= 1/T)

    loss_history, flow_model, score_model = train_with_fmot_scorematching(
        func_v=func_v,
        func_s=score_net,
        adatas_list=adatas, 
        otfms=otfms, 
        cond_matrix=conditionals,
        alpha=0.1,
        reg=5e-6,
        n_steps=15000,
        batch_size=batch_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
        lr=3e-3,
        true_mat = true_matrix
    )

    def maskdiag(A):
        return A * (1 - np.eye(n)) 

    def compute_global_jacobian(v, adatas, dt, device=torch.device("cpu")):
        """
        Compute a single adjacency from a big set of states across all datasets.
        Returns a [d, d] numpy array representing an average Jacobian.
        """

        all_x_list = []
        for ds_idx, adata in enumerate(adatas):
            x0 = adata.X[adata.obs["t"] == 0]
            all_x_list.append(x0)
        if len(all_x_list) == 0:
            return None

        X_all = np.concatenate(all_x_list, axis=0)
        if X_all.shape[0] == 0:
            return None
        
        X_all_torch = torch.from_numpy(X_all).float().to(device)

        def get_flow(t, x):
            x_input = x.unsqueeze(0).unsqueeze(0) 
            t_input = t.unsqueeze(0).unsqueeze(0)  
            return v(t_input, x_input).squeeze(0).squeeze(0)

        # Or loop over multiple times if the model is time-varying
        t_val = torch.tensor(0.0).to(device)

        Ju = torch.func.jacrev(get_flow, argnums=1)

        Js = []

        batch_size = 256
        for start in range(0, X_all_torch.shape[0], batch_size):
            end = start + batch_size
            batch_x = X_all_torch[start:end]

            J_local = torch.vmap(lambda x: Ju(t_val, x))(batch_x)
            J_avg = J_local.mean(dim=0)
            Js.append(J_avg)

        if len(Js) == 0:
            return None
        J_final = torch.stack(Js, dim=0).mean(dim=0)

        A_est = J_final

        return A_est.detach().cpu().numpy().T

    with torch.no_grad():
        A_estim = compute_global_jacobian(func_v, adatas, dt=1/T, device=torch.device("cpu"))

    W_v = func_v.causal_graph(w_threshold=0.0).T 
    A_true = true_matrix


    # Display both the estimated adjacency matrix and the learned causal graph
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(maskdiag(A_estim), vmin=-0.5, vmax=0.5, cmap="RdBu_r"); plt.gca().invert_yaxis()
    plt.title("A_estim (from Jacobian)")
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.imshow(maskdiag(W_v), cmap="Reds"); plt.gca().invert_yaxis()
    plt.title("Causal Graph (from MLPODEF)")
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.imshow(maskdiag(A_true), vmin=-1, vmax=1, cmap="RdBu_r"); plt.gca().invert_yaxis()
    plt.title("A_true")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    maskdiag(W_v)

    # Compute and display precision-recall curves for both methods
    from sklearn.metrics import precision_recall_curve, average_precision_score

    plt.figure(figsize=(12, 5))
    # For Jacobian-based estimation
    plt.subplot(1, 2, 1)
    y_true = np.abs(np.sign(maskdiag(A_true)).astype(int).flatten())
    y_pred = np.abs(maskdiag(A_estim).flatten())
    prec, rec, thresh = precision_recall_curve(y_true, y_pred)
    avg_prec = average_precision_score(y_true, y_pred)
    plt.plot(rec, prec, label=f"Jacobian-based (AP = {avg_prec:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(
        f"Precision-Recall Curve (Jacobian)\nAUPR ratio = {avg_prec/np.mean(np.abs(A_true) > 0)}"
    )
    plt.legend()
    plt.grid(True)
    # For MLPODEF-based estimation
    plt.subplot(1, 2, 2)
    y_pred_mlp = np.abs(maskdiag(W_v).flatten())
    prec, rec, thresh = precision_recall_curve(y_true, y_pred_mlp)
    avg_prec_mlp = average_precision_score(y_true, y_pred_mlp)
    plt.plot(rec, prec, label=f"MLPODEF-based (AP = {avg_prec_mlp:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(
        f"Precision-Recall Curve (MLPODEF)\nAUPR ratio = {avg_prec_mlp/np.mean(np.abs(A_true) > 0)}"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    def evaluate_all_datasets(func, adatas_list, n_samples=100, device="cpu", dataset_idxs=None):
        """
        Evaluate trajectory predictions for a list of anndata objects.
        
        Parameters:
        - func: a function like plot_predicted_vs_true_with_metrics that returns metrics
        - adatas_list: list of anndata objects
        - n_samples: number of samples to use per time bin
        - device: device for computation
        - dataset_idxs: optional list of indices corresponding to each dataset (useful if you have knockout masks)
        
        Returns:
        - results: a dictionary mapping dataset index to a list of metric tuples (one per time bin).
        """
        results = {}
        for i, adata in enumerate(adatas_list):
            ds_idx = dataset_idxs[i] if dataset_idxs is not None else None
            print(f"Evaluating dataset {i} (dataset_idx = {ds_idx})")
            time_bins = np.unique(adata.obs["t"].values).astype(int)
            ds_metrics = []
            for tb in time_bins[:-1]:
                mse_val, corr_val, r2_val, wass_val = plot_predicted_vs_true_with_metrics(
                    func_v=func, adata=adata, start_bin=tb, end_bin=tb+1, n_samples=n_samples, device=device
                )
                ds_metrics.append((tb, mse_val, corr_val, r2_val, wass_val))
            results[i] = ds_metrics
        return results
    
    results = evaluate_all_datasets(
        func=flow_model, 
        adatas_list=adatas, 
        n_samples=185, 
        device="cuda" if torch.cuda.is_available() else "cpu", 
        dataset_idxs=list(range(len(adatas)))
    )

    all_mse = []
    for ds, metrics in results.items():
        for (tb, mse_val, corr_val, r2_val, wass_val) in metrics:
            all_mse.append(mse_val)
    print("Average MSE across datasets and time bins:", np.mean(all_mse))


def main_with_holdout():
    """
    Run leave-one-out evaluation for the linear model.
    """
    T = 5
    data_loader = DataLoader("../../data/simulation", dataset_type="Synthetic")
    data_loader.load_data()
    adatas, kos, ko_indices, true_matrix = (
        data_loader.adatas,
        data_loader.kos,
        data_loader.ko_indices,
        data_loader.true_matrix.values,
    )

    num_variables = 8
    results = []

    for held_out_time in range(1, T):
        print(f"\n=== Training with timepoint {held_out_time} held out ===")

        avg_distances, flow_model, score_model = train_and_evaluate_with_holdout(
            adatas=adatas, held_out_time=held_out_time, num_variables=num_variables, kos=kos, ko_indices=ko_indices, true_matrix=true_matrix
        )

        results.append({"held_out_time": held_out_time, "distances": avg_distances})

        print(f"Results for held-out timepoint {held_out_time}:")
        print(f"ODE distance: {avg_distances['ode']:.4f}")
        print(f"SDE distance: {avg_distances['sde']:.4f}")

    return results


if __name__ == "__main__":
    #main_with_holdout()
    main()