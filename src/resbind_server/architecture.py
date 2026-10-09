from __future__ import annotations
import torch
from torch import nn

TASKS = ("metal", "nuclear", "small")

DEFAULT_INITIAL_GATES = (
    0.90, 0.20, 0.20,
    0.90, 0.70, 0.20,
    0.90, 0.70, 0.70,
)

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
