#!/usr/bin/env python
"""Train a unified multi-task shallow-shared gated MLP model.

V6 shallow-shared gated MLP:
  - Frozen ProtT5 residue embeddings are used directly.
  - Surface descriptors and contact-neighbor summaries use separate encoders.
  - The three view encoders are shared across tasks.
  - Each task has its own independent sigmoid gate and MLP head.
  - No deep shared fusion layer is used after gating, reducing negative transfer.
  - Missing task labels are masked out in the BCE loss and metrics.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, matthews_corrcoef, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn


TASKS = ("metal", "nuclear", "small")
SPLITS = ("train", "val", "test", "external_test")
BIDIRECTIONAL_CONTACTS = True
DEFAULT_INITIAL_GATES = (
    0.90, 0.20, 0.20,
    0.90, 0.70, 0.20,
    0.90, 0.70, 0.70,
)

SURFACE_COLUMNS = [
    "protein_length",
    "distance_to_n_term",
    "distance_to_c_term",
    "relative_position",
    "is_n_terminal_window",
    "is_c_terminal_window",
    "center_rsa",
    "center_exposed",
    "center_asa",
    "center_disorder",
    "center_p_helix",
    "center_p_strand",
    "center_p_coil",
    "window_mean_rsa",
    "window_max_rsa",
    "window_min_rsa",
    "window_std_rsa",
    "window_exposed_ratio",
    "window_mean_asa",
    "window_max_asa",
    "window_mean_disorder",
    "window_max_disorder",
    "window_mean_p_helix",
    "window_mean_p_strand",
    "window_mean_p_coil",
]


@dataclass
class ProteinGraph:
    peptide_id: str
    prot: torch.Tensor
    surface: torch.Tensor
    contact: torch.Tensor
    y: torch.Tensor
    mask: torch.Tensor
    meta: pd.DataFrame


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ViewEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
        )
        self.residual = ResidualBlock(out_dim, hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual(self.proj(x))


class MultitaskShallowSharedGatedMLP(nn.Module):
    def __init__(
        self,
        prot_dim: int,
        surface_dim: int,
        contact_dim: int,
        view_dim: int,
        view_hidden_dim: int,
        classifier_hidden_dim: int,
        dropout: float,
        n_tasks: int,
        disabled_views: tuple[str, ...] = (),
        gate_mode: str = "learned",
        sharing_mode: str = "shared",
        initial_gates: tuple[float, ...] = DEFAULT_INITIAL_GATES,
    ):
        super().__init__()
        self.n_tasks = n_tasks
        self.disabled_views = tuple(disabled_views)
        self.gate_mode = gate_mode
        self.sharing_mode = sharing_mode
        if sharing_mode == "shared":
            self.prot_encoder = ViewEncoder(prot_dim, view_dim, view_hidden_dim, dropout)
            self.surface_encoder = ViewEncoder(surface_dim, view_dim, view_hidden_dim, dropout)
            self.contact_encoder = ViewEncoder(contact_dim, view_dim, view_hidden_dim, dropout)
        else:
            self.prot_encoders = nn.ModuleList(
                [ViewEncoder(prot_dim, view_dim, view_hidden_dim, dropout) for _ in range(n_tasks)]
            )
            self.surface_encoders = nn.ModuleList(
                [ViewEncoder(surface_dim, view_dim, view_hidden_dim, dropout) for _ in range(n_tasks)]
            )
            self.contact_encoders = nn.ModuleList(
                [ViewEncoder(contact_dim, view_dim, view_hidden_dim, dropout) for _ in range(n_tasks)]
            )
        expected_gate_count = n_tasks * 3
        if len(initial_gates) != expected_gate_count:
            raise ValueError(f"Expected {expected_gate_count} initial gate values, got {len(initial_gates)}")
        initial_gate_tensor = torch.tensor(initial_gates, dtype=torch.float32).reshape(n_tasks, 3)
        if not torch.all((initial_gate_tensor > 0.0) & (initial_gate_tensor < 1.0)):
            raise ValueError("Initial gates must be strictly between 0 and 1")
        self.task_gate_logits = nn.Parameter(torch.logit(initial_gate_tensor))
        if gate_mode == "fixed_initial":
            self.task_gate_logits.requires_grad_(False)
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(view_dim),
                    nn.Linear(view_dim, classifier_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(classifier_hidden_dim, 1),
                )
                for _ in range(n_tasks)
            ]
        )

    def _masked_views(self, views: torch.Tensor) -> torch.Tensor:
        if not self.disabled_views:
            return views
        views = views.clone()
        view_to_index = {"prot": 0, "surface": 1, "contact": 2}
        for view in self.disabled_views:
            views[:, view_to_index[view], :] = 0.0
        return views

    def _gate_weights_tensor(self) -> torch.Tensor:
        if self.gate_mode == "equal":
            return torch.ones(
                (self.n_tasks, 3),
                dtype=self.task_gate_logits.dtype,
                device=self.task_gate_logits.device,
            )
        return torch.sigmoid(self.task_gate_logits)

    def _shared_views(self, prot: torch.Tensor, surface: torch.Tensor, contact: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                self.prot_encoder(prot),
                self.surface_encoder(surface),
                self.contact_encoder(contact),
            ],
            dim=1,
        )

    def _private_views(
        self,
        task_index: int,
        prot: torch.Tensor,
        surface: torch.Tensor,
        contact: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack(
            [
                self.prot_encoders[task_index](prot),
                self.surface_encoders[task_index](surface),
                self.contact_encoders[task_index](contact),
            ],
            dim=1,
        )

    def forward(self, prot: torch.Tensor, surface: torch.Tensor, contact: torch.Tensor) -> torch.Tensor:
        gates = self._gate_weights_tensor()
        shared_views = self._masked_views(self._shared_views(prot, surface, contact)) if self.sharing_mode == "shared" else None
        logits = []
        for task_index in range(self.n_tasks):
            if shared_views is None:
                views = self._masked_views(self._private_views(task_index, prot, surface, contact))
            else:
                views = shared_views
            fused = (views * gates[task_index].view(1, 3, 1)).sum(dim=1)
            logits.append(self.heads[task_index](fused).squeeze(1))
        return torch.stack(logits, dim=1)

    @torch.no_grad()
    def gate_weights(self) -> dict[str, list[float]]:
        gates = self._gate_weights_tensor().detach().cpu().numpy()
        return {task: gates[i].round(6).tolist() for i, task in enumerate(TASKS)}

    def gate_sparsity_loss(self) -> torch.Tensor:
        if self.gate_mode == "equal":
            return torch.zeros((), dtype=self.task_gate_logits.dtype, device=self.task_gate_logits.device)
        gates = torch.sigmoid(self.task_gate_logits)
        return gates[:, 1:].mean()

    def shared_parameters_for_gradnorm(self):
        if self.sharing_mode == "shared":
            for module in (self.prot_encoder, self.surface_encoder, self.contact_encoder):
                yield from module.parameters()
        else:
            for modules in (self.prot_encoders, self.surface_encoders, self.contact_encoders):
                for module in modules:
                    yield from module.parameters()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--contact-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--view-dim", type=int, default=256)
    parser.add_argument("--view-hidden-dim", type=int, default=256)
    parser.add_argument("--classifier-hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gate-l1", type=float, default=1e-5)
    parser.add_argument(
        "--initial-gates",
        type=float,
        nargs=9,
        default=list(DEFAULT_INITIAL_GATES),
        metavar=(
            "METAL_PROT", "METAL_SURFACE", "METAL_CONTACT",
            "NUCLEAR_PROT", "NUCLEAR_SURFACE", "NUCLEAR_CONTACT",
            "SMALL_PROT", "SMALL_SURFACE", "SMALL_CONTACT",
        ),
        help="Initial sigmoid gates in task order metal, nuclear, small and view order prot, surface, contact.",
    )
    parser.add_argument("--loss-balancing", choices=["none", "gradnorm"], default="none")
    parser.add_argument("--gradnorm-alpha", type=float, default=1.5)
    parser.add_argument("--gradnorm-lr", type=float, default=1e-3)
    parser.add_argument("--disabled-views", nargs="*", choices=["prot", "surface", "contact"], default=[])
    parser.add_argument("--gate-mode", choices=["learned", "equal", "fixed_initial"], default="learned")
    parser.add_argument("--sharing-mode", choices=["shared", "task_private"], default="shared")
    parser.add_argument("--ablation-name", default="full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-proteins-per-split", type=int, default=None)
    parser.add_argument("--bidirectional-contacts", action="store_true", default=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_task_features(feature_root: Path, split: str, task: str) -> pd.DataFrame:
    path = feature_root / split / f"{task}_full_protein_refinement_features.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    frame["peptide_id"] = frame["peptide_id"].astype(str)
    frame["center_pos"] = frame["center_pos"].astype(int)
    frame["label"] = frame["label"].astype(int)
    frame["task"] = task
    return frame


def read_split_frames(feature_root: Path, split: str) -> dict[str, pd.DataFrame]:
    return {task: read_task_features(feature_root, split, task) for task in TASKS}


def load_embedding(embedding_dir: Path, protein_id: str, cache: dict[str, np.ndarray]) -> np.ndarray | None:
    if protein_id in cache:
        return cache[protein_id]
    path = embedding_dir / f"{protein_id}.npy"
    if not path.exists():
        return None
    arr = np.load(path).astype(np.float32, copy=False)
    if arr.ndim != 2:
        return None
    cache[protein_id] = arr
    return arr


def read_contact_edges(contact_dir: Path, protein_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    path = contact_dir / f"{protein_id}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    required = {"residue_number", "edge_i", "edge_j"}
    if not required.issubset(data.files):
        return None
    return data["residue_number"].astype(int), data["edge_i"].astype(int), data["edge_j"].astype(int)


def make_edges_bidirectional(edge_i: np.ndarray, edge_j: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pairs = np.concatenate((np.column_stack((edge_i, edge_j)), np.column_stack((edge_j, edge_i))))
    pairs = np.unique(pairs[pairs[:, 0] != pairs[:, 1]], axis=0)
    return pairs[:, 0], pairs[:, 1]


def surface_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return frame[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)


def first_feature_rows(task_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged = pd.concat(task_frames.values(), ignore_index=True)
    merged = merged.sort_values(["peptide_id", "center_pos", "task"])
    return merged.drop_duplicates(["peptide_id", "center_pos"], keep="first").copy()


def collect_surface_scaler(feature_root: Path, surface_columns: list[str]) -> StandardScaler:
    train_frames = read_split_frames(feature_root, "train")
    train_unique = first_feature_rows(train_frames)
    return StandardScaler().fit(surface_matrix(train_unique, surface_columns))


def pool_neighbors(values: np.ndarray, edge_i: np.ndarray, edge_j: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pooled = np.zeros_like(values, dtype=np.float32)
    degree = np.zeros(values.shape[0], dtype=np.float32)
    if len(edge_i):
        np.add.at(pooled, edge_i, values[edge_j])
        np.add.at(degree, edge_i, 1.0)
    pooled = pooled / np.maximum(degree[:, None], 1.0)
    return pooled, degree


def build_contact_features_for_rows(
    group: pd.DataFrame,
    prot: np.ndarray,
    surface: np.ndarray,
    contact_dir: Path,
    protein_id: str,
) -> np.ndarray:
    contact_dim = prot.shape[1] + surface.shape[1] + 2
    contact = read_contact_edges(contact_dir, protein_id)
    if contact is None:
        return np.zeros((len(group), contact_dim), dtype=np.float32)
    residue_numbers, edge_i, edge_j = contact
    if BIDIRECTIONAL_CONTACTS:
        edge_i, edge_j = make_edges_bidirectional(edge_i, edge_j)
    contact_len = len(residue_numbers)
    contact_prot = np.zeros((contact_len, prot.shape[1]), dtype=np.float32)
    contact_surface = np.zeros((contact_len, surface.shape[1]), dtype=np.float32)
    row_positions = group["center_pos"].astype(int).to_numpy()
    pos_to_row = {int(pos): idx for idx, pos in enumerate(row_positions)}
    residue_to_contact_index = {}
    for contact_idx, residue_number in enumerate(residue_numbers):
        residue = int(residue_number)
        residue_to_contact_index[residue] = int(contact_idx)
        row_idx = pos_to_row.get(residue)
        if row_idx is not None:
            contact_prot[contact_idx] = prot[row_idx]
            contact_surface[contact_idx] = surface[row_idx]
    pooled_prot, degree = pool_neighbors(contact_prot, edge_i, edge_j)
    pooled_surface, _ = pool_neighbors(contact_surface, edge_i, edge_j)
    rows = []
    for pos in row_positions:
        contact_idx = residue_to_contact_index.get(int(pos))
        if contact_idx is None:
            rows.append(np.zeros(contact_dim, dtype=np.float32))
            continue
        rows.append(
            np.concatenate(
                [
                    pooled_prot[contact_idx],
                    pooled_surface[contact_idx],
                    np.array([degree[contact_idx], 1.0], dtype=np.float32),
                ]
            )
        )
    return np.vstack(rows).astype(np.float32)


def build_multitask_frame(task_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = first_feature_rows(task_frames)
    label_parts = []
    for task in TASKS:
        task_labels = (
            task_frames[task]
            .groupby(["peptide_id", "center_pos"], as_index=False)["label"]
            .max()
            .rename(columns={"label": f"label_{task}"})
        )
        label_parts.append(task_labels)
    out = base.copy()
    for labels in label_parts:
        out = out.merge(labels, on=["peptide_id", "center_pos"], how="left")
    for task in TASKS:
        out[f"mask_{task}"] = out[f"label_{task}"].notna().astype(np.float32)
        out[f"label_{task}"] = out[f"label_{task}"].fillna(0).astype(np.float32)
    return out


def build_graphs(
    frame: pd.DataFrame,
    surface_columns: list[str],
    surface_scaler: StandardScaler,
    embedding_dir: Path,
    contact_dir: Path,
    embedding_cache: dict[str, np.ndarray],
    max_proteins: int | None,
) -> tuple[list[ProteinGraph], dict[str, int]]:
    graphs = []
    skipped = {"missing_embedding_rows": 0, "out_of_range_rows": 0}
    for index, (protein_id, group) in enumerate(frame.groupby("peptide_id", sort=False)):
        if max_proteins is not None and index >= max_proteins:
            break
        emb = load_embedding(embedding_dir, protein_id, embedding_cache)
        if emb is None:
            skipped["missing_embedding_rows"] += len(group)
            continue
        group = group.sort_values("center_pos").copy()
        pos = group["center_pos"].astype(int).to_numpy()
        keep = (pos >= 1) & (pos <= emb.shape[0])
        if not np.all(keep):
            skipped["out_of_range_rows"] += int((~keep).sum())
            group = group.loc[keep].copy()
            pos = group["center_pos"].astype(int).to_numpy()
        if len(group) == 0:
            continue
        prot = emb[pos - 1]
        surface = surface_scaler.transform(surface_matrix(group, surface_columns)).astype(np.float32)
        contact = build_contact_features_for_rows(group, prot, surface, contact_dir, protein_id)
        y = group[[f"label_{task}" for task in TASKS]].to_numpy(dtype=np.float32)
        mask = group[[f"mask_{task}" for task in TASKS]].to_numpy(dtype=np.float32)
        meta_cols = ["peptide_id", "center_pos"] + [f"label_{task}" for task in TASKS] + [f"mask_{task}" for task in TASKS]
        graphs.append(
            ProteinGraph(
                peptide_id=str(protein_id),
                prot=torch.from_numpy(prot).float(),
                surface=torch.from_numpy(surface).float(),
                contact=torch.from_numpy(contact).float(),
                y=torch.from_numpy(y).float(),
                mask=torch.from_numpy(mask).float(),
                meta=group[meta_cols].copy(),
            )
        )
    return graphs, skipped


def fit_view_scalers(train_graphs: list[ProteinGraph]) -> dict[str, StandardScaler]:
    scalers = {
        "prot": StandardScaler(),
        "surface": StandardScaler(),
        "contact": StandardScaler(),
    }
    scalers["prot"].fit(np.concatenate([graph.prot.numpy() for graph in train_graphs], axis=0))
    scalers["surface"].fit(np.concatenate([graph.surface.numpy() for graph in train_graphs], axis=0))
    scalers["contact"].fit(np.concatenate([graph.contact.numpy() for graph in train_graphs], axis=0))
    return scalers


def apply_view_scalers(graphs: list[ProteinGraph], scalers: dict[str, StandardScaler]) -> None:
    for graph in graphs:
        graph.prot = torch.from_numpy(scalers["prot"].transform(graph.prot.numpy()).astype(np.float32)).float()
        graph.surface = torch.from_numpy(scalers["surface"].transform(graph.surface.numpy()).astype(np.float32)).float()
        graph.contact = torch.from_numpy(scalers["contact"].transform(graph.contact.numpy()).astype(np.float32)).float()


def masked_task_losses(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight, reduction="none")
    loss = loss * mask
    return loss.sum(dim=0) / mask.sum(dim=0).clamp_min(1.0)


def train_epoch(
    model,
    graphs: list[ProteinGraph],
    optimizer,
    weight_optimizer,
    task_log_weights: torch.Tensor | None,
    initial_task_losses: torch.Tensor | None,
    pos_weight: torch.Tensor,
    device: torch.device,
    gate_l1: float,
    loss_balancing: str,
    gradnorm_alpha: float,
) -> float:
    model.train()
    order = np.random.permutation(len(graphs))
    total_loss = 0.0
    total_mask = 0.0
    for idx in order:
        graph = graphs[int(idx)]
        optimizer.zero_grad(set_to_none=True)
        logits = model(graph.prot.to(device), graph.surface.to(device), graph.contact.to(device))
        y = graph.y.to(device)
        mask = graph.mask.to(device)
        task_losses = masked_task_losses(logits, y, mask, pos_weight)
        active = mask.sum(dim=0) > 0
        if loss_balancing == "gradnorm":
            if task_log_weights is None or initial_task_losses is None or weight_optimizer is None:
                raise RuntimeError("GradNorm requested without task weights/initial losses.")
            task_weights = torch.exp(task_log_weights)
            weighted_task_losses = task_weights * task_losses
            loss = weighted_task_losses[active].sum()
            shared_params = [param for param in model.shared_parameters_for_gradnorm() if param.requires_grad]
            grad_norms = []
            for task_index in range(len(TASKS)):
                if not bool(active[task_index]):
                    grad_norms.append(torch.zeros((), dtype=loss.dtype, device=device))
                    continue
                grads = torch.autograd.grad(
                    weighted_task_losses[task_index],
                    shared_params,
                    retain_graph=True,
                    create_graph=True,
                    allow_unused=True,
                )
                norm_terms = [grad.norm(2) for grad in grads if grad is not None]
                grad_norms.append(torch.stack(norm_terms).sum() if norm_terms else torch.zeros((), dtype=loss.dtype, device=device))
            grad_norms_t = torch.stack(grad_norms)
            with torch.no_grad():
                loss_ratio = task_losses.detach() / initial_task_losses.clamp_min(1e-8)
                active_ratio = loss_ratio[active] / loss_ratio[active].mean().clamp_min(1e-8)
                target = grad_norms_t[active].detach().mean() * (active_ratio ** gradnorm_alpha)
            gradnorm_loss = torch.abs(grad_norms_t[active] - target).sum()
            weight_optimizer.zero_grad(set_to_none=True)
            gradnorm_loss.backward(retain_graph=True)
            weight_optimizer.step()
            with torch.no_grad():
                task_log_weights -= torch.log(torch.exp(task_log_weights).mean().clamp_min(1e-8))
            optimizer.zero_grad(set_to_none=True)
            task_weights = torch.exp(task_log_weights.detach())
            loss = (task_weights * task_losses)[active].sum()
        else:
            loss = task_losses[active].sum()
        if gate_l1 > 0:
            loss = loss + gate_l1 * model.gate_sparsity_loss()
        loss.backward()
        optimizer.step()
        mask_count = float(mask.sum().detach().cpu())
        total_loss += float(loss.detach().cpu()) * mask_count
        total_mask += mask_count
    return total_loss / max(total_mask, 1.0)


@torch.no_grad()
def predict(model, graphs: list[ProteinGraph], device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    model.eval()
    probs = []
    labels = []
    masks = []
    meta = []
    for graph in graphs:
        prob = torch.sigmoid(model(graph.prot.to(device), graph.surface.to(device), graph.contact.to(device)))
        probs.append(prob.cpu().numpy())
        labels.append(graph.y.numpy())
        masks.append(graph.mask.numpy())
        meta.append(graph.meta)
    return np.concatenate(labels), np.concatenate(probs), np.concatenate(masks), pd.concat(meta, ignore_index=True)


def choose_threshold(y: np.ndarray, prob: np.ndarray, mode: str) -> float:
    best_threshold = 0.5
    best_score = -1.0
    finite_prob = prob[np.isfinite(prob)]
    if len(finite_prob):
        quantiles = np.quantile(finite_prob, np.linspace(0.01, 0.99, 99))
    else:
        quantiles = np.array([], dtype=np.float32)
    thresholds = np.unique(np.concatenate([np.linspace(0.01, 0.99, 99), quantiles]))
    for threshold in thresholds:
        pred = prob >= threshold
        precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        if mode == "mcc":
            score = matthews_corrcoef(y, pred)
        elif mode == "f2":
            denom = 4 * precision + recall
            score = (5 * precision * recall / denom) if denom > 0 else 0.0
        else:
            score = f1
        if score > best_score:
            best_threshold = float(threshold)
            best_score = float(score)
    return best_threshold


def metric_record(split: str, task: str, threshold_mode: str, threshold: float, y: np.ndarray, prob: np.ndarray) -> dict:
    pred = prob >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return {
        "split": split,
        "task": task,
        "threshold_mode": threshold_mode,
        "threshold": float(threshold),
        "n_rows": int(len(y)),
        "positives": int(y.sum()),
        "AUROC": float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else np.nan,
        "AP_AUPRC": float(average_precision_score(y, prob)) if len(np.unique(y)) == 2 else np.nan,
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "MCC": float(matthews_corrcoef(y, pred)) if len(np.unique(y)) == 2 else 0.0,
        "n_pred_positive": int(pred.sum()),
    }


def task_arrays(y: np.ndarray, prob: np.ndarray, mask: np.ndarray, task_index: int) -> tuple[np.ndarray, np.ndarray]:
    keep = mask[:, task_index] > 0
    return y[keep, task_index].astype(int), prob[keep, task_index]


def macro_ap(y: np.ndarray, prob: np.ndarray, mask: np.ndarray) -> float:
    scores = []
    for task_index in range(len(TASKS)):
        task_y, task_prob = task_arrays(y, prob, mask, task_index)
        if len(task_y) and len(np.unique(task_y)) == 2:
            scores.append(float(average_precision_score(task_y, task_prob)))
    return float(np.mean(scores)) if scores else float("nan")


@torch.no_grad()
def estimate_initial_task_losses(
    model,
    graphs: list[ProteinGraph],
    pos_weight: torch.Tensor,
    device: torch.device,
    max_graphs: int = 64,
) -> torch.Tensor:
    model.eval()
    numer = torch.zeros(len(TASKS), dtype=torch.float32, device=device)
    denom = torch.zeros(len(TASKS), dtype=torch.float32, device=device)
    for graph in graphs[:max_graphs]:
        logits = model(graph.prot.to(device), graph.surface.to(device), graph.contact.to(device))
        y = graph.y.to(device)
        mask = graph.mask.to(device)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight, reduction="none") * mask
        numer += loss.sum(dim=0)
        denom += mask.sum(dim=0)
    return numer / denom.clamp_min(1.0)


def jsonable_args(args: argparse.Namespace) -> dict:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def main() -> None:
    global BIDIRECTIONAL_CONTACTS
    args = parse_args()
    BIDIRECTIONAL_CONTACTS = bool(args.bidirectional_contacts)
    set_seed(args.seed)

    run_dir = args.out_dir / args.loss_balancing / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_frames = read_split_frames(args.feature_root, "train")
    surface_columns = [col for col in SURFACE_COLUMNS if col in next(iter(train_frames.values())).columns]
    if len(surface_columns) != len(SURFACE_COLUMNS):
        missing = sorted(set(SURFACE_COLUMNS) - set(surface_columns))
        raise RuntimeError(f"Missing surface columns: {missing}")
    surface_scaler = collect_surface_scaler(args.feature_root, surface_columns)

    embedding_cache: dict[str, np.ndarray] = {}
    graphs = {}
    skipped = {}
    raw_task_rows = {}
    for split in SPLITS:
        split_dir = args.feature_root / split
        if not split_dir.exists():
            continue
        task_frames = read_split_frames(args.feature_root, split)
        raw_task_rows[split] = {task: int(len(frame)) for task, frame in task_frames.items()}
        multitask_frame = build_multitask_frame(task_frames)
        split_graphs, split_skipped = build_graphs(
            multitask_frame,
            surface_columns,
            surface_scaler,
            args.embedding_dir,
            args.contact_dir,
            embedding_cache,
            args.max_proteins_per_split,
        )
        graphs[split] = split_graphs
        skipped[split] = split_skipped
        rows = sum(len(graph.y) for graph in split_graphs)
        supervised = (
            np.sum([graph.mask.numpy().sum(axis=0) for graph in split_graphs], axis=0)
            if split_graphs
            else np.zeros(len(TASKS))
        )
        positives = (
            np.sum([(graph.y.numpy() * graph.mask.numpy()).sum(axis=0) for graph in split_graphs], axis=0)
            if split_graphs
            else np.zeros(len(TASKS))
        )
        print(
            f"[build] split={split} proteins={len(split_graphs)} union_rows={rows} "
            f"supervised={dict(zip(TASKS, supervised.astype(int)))} positives={dict(zip(TASKS, positives.astype(int)))} "
            f"skipped={split_skipped}",
            flush=True,
        )

    for required in ("train", "val", "test"):
        if required not in graphs or not graphs[required]:
            raise RuntimeError(f"Missing required split: {required}")

    view_scalers = fit_view_scalers(graphs["train"])
    for split_graphs in graphs.values():
        apply_view_scalers(split_graphs, view_scalers)

    prot_dim = int(graphs["train"][0].prot.shape[1])
    surface_dim = int(graphs["train"][0].surface.shape[1])
    contact_dim = int(graphs["train"][0].contact.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultitaskShallowSharedGatedMLP(
        prot_dim=prot_dim,
        surface_dim=surface_dim,
        contact_dim=contact_dim,
        view_dim=args.view_dim,
        view_hidden_dim=args.view_hidden_dim,
        classifier_hidden_dim=args.classifier_hidden_dim,
        dropout=args.dropout,
        n_tasks=len(TASKS),
        disabled_views=tuple(args.disabled_views),
        gate_mode=args.gate_mode,
        sharing_mode=args.sharing_mode,
        initial_gates=tuple(args.initial_gates),
    ).to(device)

    train_y = np.concatenate([graph.y.numpy() for graph in graphs["train"]], axis=0)
    train_mask = np.concatenate([graph.mask.numpy() for graph in graphs["train"]], axis=0)
    train_pos = (train_y * train_mask).sum(axis=0)
    train_total = train_mask.sum(axis=0)
    pos_weight_np = (train_total - train_pos) / np.maximum(train_pos, 1.0)
    pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    task_log_weights = None
    weight_optimizer = None
    initial_task_losses = None
    if args.loss_balancing == "gradnorm":
        task_log_weights = nn.Parameter(torch.zeros(len(TASKS), dtype=torch.float32, device=device))
        weight_optimizer = torch.optim.Adam([task_log_weights], lr=args.gradnorm_lr)
        initial_task_losses = estimate_initial_task_losses(model, graphs["train"], pos_weight, device).detach()

    print(
        f"[model] device={device} prot_dim={prot_dim} surface_dim={surface_dim} contact_dim={contact_dim} "
        f"view_dim={args.view_dim} output_dim={len(TASKS)} loss_balancing={args.loss_balancing} "
        f"disabled_views={args.disabled_views} gate_mode={args.gate_mode} sharing_mode={args.sharing_mode} "
        f"pos_weight={dict(zip(TASKS, pos_weight_np.round(4)))}",
        flush=True,
    )

    best_state = None
    best_val_macro_ap = -1.0
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            graphs["train"],
            optimizer,
            weight_optimizer,
            task_log_weights,
            initial_task_losses,
            pos_weight,
            device,
            args.gate_l1,
            args.loss_balancing,
            args.gradnorm_alpha,
        )
        val_y, val_prob, val_mask, _ = predict(model, graphs["val"], device)
        val_macro_ap = macro_ap(val_y, val_prob, val_mask)
        row = {"epoch": epoch, "train_loss": train_loss, "val_macro_ap": val_macro_ap}
        if task_log_weights is not None:
            weights = torch.exp(task_log_weights.detach().cpu()).numpy()
            row.update({f"task_weight_{task}": float(weights[i]) for i, task in enumerate(TASKS)})
        history.append(row)
        print(f"[epoch {epoch:03d}] train_loss={train_loss:.6f} val_macro_ap={val_macro_ap:.6f}", flush=True)
        if val_macro_ap > best_val_macro_ap:
            best_val_macro_ap = float(val_macro_ap)
            best_epoch = epoch
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_y, val_prob, val_mask, _ = predict(model, graphs["val"], device)
    thresholds: dict[str, dict[str, float]] = {}
    for task_index, task in enumerate(TASKS):
        task_y, task_prob = task_arrays(val_y, val_prob, val_mask, task_index)
        thresholds[task] = {
            "fixed_0p5": 0.5,
            "val_mcc": choose_threshold(task_y, task_prob, "mcc"),
            "val_f1": choose_threshold(task_y, task_prob, "f1"),
            "val_f2_recall_oriented": choose_threshold(task_y, task_prob, "f2"),
        }

    metrics = []
    for split in [key for key in SPLITS if key in graphs and key != "train"]:
        y, prob, mask, meta = predict(model, graphs[split], device)
        pred_frame = meta.copy()
        for task_index, task in enumerate(TASKS):
            pred_frame[f"prob_{task}"] = prob[:, task_index]
        pred_frame.to_csv(run_dir / f"{split}_predictions.csv", index=False)
        for task_index, task in enumerate(TASKS):
            task_y, task_prob = task_arrays(y, prob, mask, task_index)
            for threshold_mode, threshold in thresholds[task].items():
                rec = metric_record(split, task, threshold_mode, threshold, task_y, task_prob)
                rec.update({"model_variant": args.ablation_name})
                metrics.append(rec)

    pd.DataFrame(metrics).to_csv(run_dir / "metrics.csv", index=False)
    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)
    torch.save({"model_state_dict": model.state_dict(), "args": jsonable_args(args), "tasks": TASKS}, run_dir / "model.pt")

    summary = {
        "model_variant": args.ablation_name,
        "tasks": TASKS,
        "device": str(device),
        "prot_dim": prot_dim,
        "surface_dim": surface_dim,
        "contact_dim": contact_dim,
        "view_dim": args.view_dim,
        "view_hidden_dim": args.view_hidden_dim,
        "classifier_hidden_dim": args.classifier_hidden_dim,
        "output_dim": len(TASKS),
        "best_epoch": best_epoch,
        "best_val_macro_ap": best_val_macro_ap,
        "gate_order": ["prot", "surface", "contact"],
        "gate_type": args.gate_mode,
        "initial_gate_weights": {
            task: [float(value) for value in args.initial_gates[index * 3 : (index + 1) * 3]]
            for index, task in enumerate(TASKS)
        },
        "sharing_mode": args.sharing_mode,
        "gate_l1": args.gate_l1,
        "bidirectional_contacts": BIDIRECTIONAL_CONTACTS,
        "gate_weights": model.gate_weights(),
        "disabled_views": args.disabled_views,
        "loss_balancing": args.loss_balancing,
        "gradnorm_alpha": args.gradnorm_alpha,
        "initial_task_losses": {
            task: float(initial_task_losses.detach().cpu()[i]) for i, task in enumerate(TASKS)
        }
        if initial_task_losses is not None
        else None,
        "task_loss_weights": {
            task: float(torch.exp(task_log_weights.detach().cpu())[i]) for i, task in enumerate(TASKS)
        }
        if task_log_weights is not None
        else None,
        "thresholds": thresholds,
        "raw_task_rows": raw_task_rows,
        "supervised_rows": {
            split: {
                task: int(sum(graph.mask[:, task_index].sum().item() for graph in split_graphs))
                for task_index, task in enumerate(TASKS)
            }
            for split, split_graphs in graphs.items()
        },
        "positives": {
            split: {
                task: int(sum((graph.y[:, task_index] * graph.mask[:, task_index]).sum().item() for graph in split_graphs))
                for task_index, task in enumerate(TASKS)
            }
            for split, split_graphs in graphs.items()
        },
        "skipped": skipped,
        "surface_columns": surface_columns,
        "config": jsonable_args(args),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)
    print(f"Wrote {run_dir}", flush=True)


if __name__ == "__main__":
    main()
