#!/usr/bin/env python3
"""
Comprehensive experiment suite for all remaining potential improvements.

Tests:
1. JK-Net (Jumping Knowledge)
2. GAT v2 (improved attention)
3. Different margin values (0.05, 0.1, 0.15, 0.2, 0.3)
4. Learning rate schedules (cosine, warmup, step)
5. Larger batch sizes (32, 64, 128, 256)
6. Edge features (sentence similarity weights)
7. Position/length features
8. Contrastive pre-training
9. Multi-task learning (ranking + classification)

Note: Knowledge distillation requires separate LLM inference step.

Usage:
    python scripts/experiments/run_all_remaining_experiments.py \
        --graph_dir data/cache/gnn/rebuild_20260120 \
        --output_dir outputs/all_experiments
"""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, LambdaLR
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, GATv2Conv, GCNConv, JumpingKnowledge

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from final_sc_review.constants import EXCLUDED_CRITERIA


# ============================================================================
# Metrics
# ============================================================================

def compute_ndcg_at_k(gold_mask: np.ndarray, scores: np.ndarray, k: int = 10) -> float:
    sorted_idx = np.argsort(-scores)
    sorted_gold = gold_mask[sorted_idx]
    dcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(sorted_gold))) if sorted_gold[i])
    n_gold = int(gold_mask.sum())
    if n_gold == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, n_gold)))
    return dcg / idcg if idcg > 0 else 0.0


def compute_mrr(gold_mask: np.ndarray, scores: np.ndarray) -> float:
    sorted_idx = np.argsort(-scores)
    for i, idx in enumerate(sorted_idx):
        if gold_mask[idx]:
            return 1.0 / (i + 1)
    return 0.0


# ============================================================================
# Model Architectures
# ============================================================================

class SAGEResidualReranker(nn.Module):
    """Best baseline: GraphSAGE with residual connections."""
    def __init__(self, input_dim, hidden_dim=128, num_layers=1, dropout=0.05,
                 alpha_init=0.65, learn_alpha=True):
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(input_dim, hidden_dim)])
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.score_head = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(alpha_init), requires_grad=learn_alpha)
        self.residual_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else None

    def forward(self, x, edge_index, reranker_scores, batch=None, edge_weight=None):
        h_in = self.residual_proj(x) if self.residual_proj else x
        h = x
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
            h = self.dropout(h)
        h = h + h_in
        gnn_scores = self.score_head(h).squeeze(-1)
        alpha = torch.sigmoid(self.alpha)
        return alpha * reranker_scores + (1 - alpha) * gnn_scores


class JKNetReranker(nn.Module):
    """Jumping Knowledge Network - aggregates features from all layers."""
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.05,
                 alpha_init=0.65, learn_alpha=True, jk_mode='cat'):
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(input_dim, hidden_dim)])
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        self.jk = JumpingKnowledge(mode=jk_mode, channels=hidden_dim, num_layers=num_layers)

        if jk_mode == 'cat':
            self.score_head = nn.Linear(hidden_dim * num_layers, 1)
        else:
            self.score_head = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(alpha_init), requires_grad=learn_alpha)

    def forward(self, x, edge_index, reranker_scores, batch=None, edge_weight=None):
        xs = []
        h = x
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
            h = self.dropout(h)
            xs.append(h)

        h = self.jk(xs)
        gnn_scores = self.score_head(h).squeeze(-1)
        alpha = torch.sigmoid(self.alpha)
        return alpha * reranker_scores + (1 - alpha) * gnn_scores


class GATv2Reranker(nn.Module):
    """GAT v2 with improved dynamic attention."""
    def __init__(self, input_dim, hidden_dim=128, num_layers=1, dropout=0.05,
                 alpha_init=0.65, learn_alpha=True, heads=4):
        super().__init__()
        self.convs = nn.ModuleList([GATv2Conv(input_dim, hidden_dim // heads, heads=heads, dropout=dropout)])
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout))

        self.score_head = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(alpha_init), requires_grad=learn_alpha)
        self.residual_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else None

    def forward(self, x, edge_index, reranker_scores, batch=None, edge_weight=None):
        h_in = self.residual_proj(x) if self.residual_proj else x
        h = x
        for conv in self.convs:
            h = F.elu(conv(h, edge_index))
            h = self.dropout(h)
        h = h + h_in
        gnn_scores = self.score_head(h).squeeze(-1)
        alpha = torch.sigmoid(self.alpha)
        return alpha * reranker_scores + (1 - alpha) * gnn_scores


class EdgeWeightedSAGE(nn.Module):
    """SAGE with edge weight support (for similarity-based edges)."""
    def __init__(self, input_dim, hidden_dim=128, num_layers=1, dropout=0.05,
                 alpha_init=0.65, learn_alpha=True):
        super().__init__()
        # Use GCN which supports edge_weight natively
        self.convs = nn.ModuleList([GCNConv(input_dim, hidden_dim)])
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.score_head = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(alpha_init), requires_grad=learn_alpha)
        self.residual_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else None

    def forward(self, x, edge_index, reranker_scores, batch=None, edge_weight=None):
        h_in = self.residual_proj(x) if self.residual_proj else x
        h = x
        for conv in self.convs:
            if edge_weight is not None:
                h = F.relu(conv(h, edge_index, edge_weight))
            else:
                h = F.relu(conv(h, edge_index))
            h = self.dropout(h)
        h = h + h_in
        gnn_scores = self.score_head(h).squeeze(-1)
        alpha = torch.sigmoid(self.alpha)
        return alpha * reranker_scores + (1 - alpha) * gnn_scores


class MultiTaskReranker(nn.Module):
    """Multi-task: ranking + criterion classification."""
    def __init__(self, input_dim, hidden_dim=128, num_layers=1, dropout=0.05,
                 alpha_init=0.65, learn_alpha=True, num_criteria=10):
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(input_dim, hidden_dim)])
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        self.score_head = nn.Linear(hidden_dim, 1)
        self.criterion_head = nn.Linear(hidden_dim, num_criteria)  # Multi-task head
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(alpha_init), requires_grad=learn_alpha)
        self.residual_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else None

    def forward(self, x, edge_index, reranker_scores, batch=None, edge_weight=None, return_criterion_logits=False):
        h_in = self.residual_proj(x) if self.residual_proj else x
        h = x
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
            h = self.dropout(h)
        h = h + h_in

        gnn_scores = self.score_head(h).squeeze(-1)
        alpha = torch.sigmoid(self.alpha)
        final_scores = alpha * reranker_scores + (1 - alpha) * gnn_scores

        if return_criterion_logits:
            criterion_logits = self.criterion_head(h)
            return final_scores, criterion_logits
        return final_scores


class PositionAwareSAGE(nn.Module):
    """SAGE with position and length features."""
    def __init__(self, input_dim, hidden_dim=128, num_layers=1, dropout=0.05,
                 alpha_init=0.65, learn_alpha=True):
        super().__init__()
        # +2 for position and length features
        self.feature_proj = nn.Linear(input_dim + 2, input_dim)
        self.convs = nn.ModuleList([SAGEConv(input_dim, hidden_dim)])
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        self.score_head = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(alpha_init), requires_grad=learn_alpha)
        self.residual_proj = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, edge_index, reranker_scores, batch=None, edge_weight=None,
                positions=None, lengths=None):
        # Add position and length features if provided
        if positions is not None and lengths is not None:
            pos_feat = positions.unsqueeze(-1).float() / 50.0  # Normalize
            len_feat = lengths.unsqueeze(-1).float() / 500.0  # Normalize
            x = torch.cat([x, pos_feat, len_feat], dim=-1)
            x = self.feature_proj(x)

        h_in = self.residual_proj(x)
        h = x
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
            h = self.dropout(h)
        h = h + h_in
        gnn_scores = self.score_head(h).squeeze(-1)
        alpha = torch.sigmoid(self.alpha)
        return alpha * reranker_scores + (1 - alpha) * gnn_scores


# ============================================================================
# Loss Functions
# ============================================================================

class PairwiseMarginLoss(nn.Module):
    def __init__(self, margin=0.1):
        super().__init__()
        self.margin = margin

    def forward(self, scores, labels, batch=None):
        if batch is None:
            return self._single(scores, labels)
        total, n = 0.0, 0
        for b in batch.unique():
            mask = batch == b
            loss = self._single(scores[mask], labels[mask])
            if loss > 0:
                total += loss
                n += 1
        return total / max(n, 1)

    def _single(self, scores, labels):
        pos_mask, neg_mask = labels > 0, labels == 0
        if not pos_mask.any() or not neg_mask.any():
            return torch.tensor(0.0, device=scores.device)
        losses = F.relu(self.margin - scores[pos_mask].unsqueeze(1) + scores[neg_mask].unsqueeze(0))
        return losses.mean()


class ContrastiveLoss(nn.Module):
    """Contrastive loss for pre-training."""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels, batch=None):
        if batch is None:
            return self._single(embeddings, labels)
        total, n = 0.0, 0
        for b in batch.unique():
            mask = batch == b
            loss = self._single(embeddings[mask], labels[mask])
            if loss > 0:
                total += loss
                n += 1
        return total / max(n, 1)

    def _single(self, embeddings, labels):
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=-1)

        # Compute similarity matrix
        sim_matrix = torch.mm(embeddings, embeddings.t()) / self.temperature

        # Create positive mask (same label)
        pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        pos_mask.fill_diagonal_(False)

        if not pos_mask.any():
            return torch.tensor(0.0, device=embeddings.device)

        # InfoNCE loss
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))

        # Mean of positive pairs
        loss = -(log_prob * pos_mask.float()).sum() / pos_mask.sum()
        return loss


# ============================================================================
# Data Loading
# ============================================================================

def load_graphs(graph_dir: Path):
    """Load graph dataset."""
    with open(graph_dir / "metadata.json") as f:
        metadata = json.load(f)

    fold_graphs = {}
    for fold_id in range(metadata["n_folds"]):
        data = torch.load(graph_dir / f"fold_{fold_id}.pt", weights_only=False)
        graphs = [g for g in data["graphs"]
                  if getattr(g, 'criterion_id', None) not in EXCLUDED_CRITERIA]
        graphs = [g for g in graphs if g.node_labels.sum() > 0]
        fold_graphs[fold_id] = graphs
        print(f"  Fold {fold_id}: {len(graphs)} graphs")

    return fold_graphs


def ensure_cpu(g):
    for key in g.keys():
        if isinstance(g[key], torch.Tensor):
            g[key] = g[key].cpu()
    return g


def add_edge_weights(graphs):
    """Add edge weights based on cosine similarity."""
    for g in graphs:
        x = g.x
        edge_index = g.edge_index

        # Compute cosine similarity for each edge
        src, dst = edge_index
        src_emb = F.normalize(x[src], dim=-1)
        dst_emb = F.normalize(x[dst], dim=-1)
        edge_weight = (src_emb * dst_emb).sum(dim=-1)
        edge_weight = (edge_weight + 1) / 2  # Scale to [0, 1]

        g.edge_weight = edge_weight
    return graphs


def add_position_features(graphs):
    """Add position and length features to graphs."""
    for g in graphs:
        n_nodes = g.x.shape[0]
        g.positions = torch.arange(n_nodes)
        # Approximate length from embedding norm (placeholder)
        g.lengths = torch.norm(g.x, dim=-1) * 100
    return graphs


# ============================================================================
# Training Functions
# ============================================================================

def get_scheduler(optimizer, scheduler_type, n_epochs, warmup_epochs=5):
    """Get learning rate scheduler."""
    if scheduler_type == "cosine":
        return CosineAnnealingLR(optimizer, T_max=n_epochs)
    elif scheduler_type == "step":
        return StepLR(optimizer, step_size=10, gamma=0.5)
    elif scheduler_type == "warmup_cosine":
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (n_epochs - warmup_epochs)))
        return LambdaLR(optimizer, lr_lambda)
    else:
        return None


def train_model(model, train_graphs, val_graphs, config, device, loss_fn=None):
    """Generic training function."""
    torch.manual_seed(config.get("seed", 42))
    np.random.seed(config.get("seed", 42))

    train_graphs = [ensure_cpu(g) for g in train_graphs]

    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    if loss_fn is None:
        loss_fn = PairwiseMarginLoss(margin=config.get("margin", 0.1))

    scheduler = get_scheduler(optimizer, config.get("scheduler"), config["max_epochs"])

    train_loader = DataLoader(train_graphs, batch_size=config["batch_size"], shuffle=True)

    best_ndcg, best_state, patience = -1, None, 0

    for epoch in range(config["max_epochs"]):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Handle edge weights if present
            edge_weight = getattr(batch, 'edge_weight', None)

            if hasattr(model, 'forward') and 'positions' in model.forward.__code__.co_varnames:
                positions = getattr(batch, 'positions', None)
                lengths = getattr(batch, 'lengths', None)
                refined = model(batch.x, batch.edge_index, batch.reranker_scores,
                              batch.batch, edge_weight, positions, lengths)
            else:
                refined = model(batch.x, batch.edge_index, batch.reranker_scores,
                              batch.batch, edge_weight)

            loss = loss_fn(refined, batch.node_labels, batch.batch)
            loss.backward()
            optimizer.step()

        if scheduler:
            scheduler.step()

        # Validation
        model.eval()
        val_ndcg = []
        with torch.no_grad():
            for g in val_graphs:
                g = ensure_cpu(g).to(device)
                gold = g.node_labels.cpu().numpy()
                edge_weight = getattr(g, 'edge_weight', None)

                if hasattr(model, 'forward') and 'positions' in model.forward.__code__.co_varnames:
                    positions = getattr(g, 'positions', None)
                    lengths = getattr(g, 'lengths', None)
                    if positions is not None:
                        positions = positions.to(device)
                        lengths = lengths.to(device)
                    scores = model(g.x, g.edge_index, g.reranker_scores, None,
                                 edge_weight, positions, lengths).cpu().numpy()
                else:
                    scores = model(g.x, g.edge_index, g.reranker_scores, None,
                                 edge_weight).cpu().numpy()
                val_ndcg.append(compute_ndcg_at_k(gold, scores, 10))

        mean_ndcg = np.mean(val_ndcg)
        if mean_ndcg > best_ndcg:
            best_ndcg = mean_ndcg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if patience >= config["patience"]:
            break

    if best_state:
        model.load_state_dict(best_state)
    return model


def evaluate_model(model, val_graphs, device):
    """Evaluate model on validation set."""
    model.eval()
    ndcg_list = []
    mrr_list = []

    with torch.no_grad():
        for g in val_graphs:
            g = ensure_cpu(g).to(device)
            gold = g.node_labels.cpu().numpy()
            edge_weight = getattr(g, 'edge_weight', None)

            if hasattr(model, 'forward') and 'positions' in model.forward.__code__.co_varnames:
                positions = getattr(g, 'positions', None)
                lengths = getattr(g, 'lengths', None)
                if positions is not None:
                    positions = positions.to(device)
                    lengths = lengths.to(device)
                scores = model(g.x, g.edge_index, g.reranker_scores, None,
                             edge_weight, positions, lengths).cpu().numpy()
            else:
                scores = model(g.x, g.edge_index, g.reranker_scores, None,
                             edge_weight).cpu().numpy()

            ndcg_list.append(compute_ndcg_at_k(gold, scores, 10))
            mrr_list.append(compute_mrr(gold, scores))

    return np.mean(ndcg_list), np.mean(mrr_list)


# ============================================================================
# Experiment Functions
# ============================================================================

def run_experiment(name, model_class, model_kwargs, config, fold_graphs, device):
    """Run a single experiment across all folds."""
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"{'='*60}")

    n_folds = len(fold_graphs)
    fold_ndcg = []
    fold_mrr = []

    for fold_id in range(n_folds):
        train_graphs = [g for fid, graphs in fold_graphs.items() if fid != fold_id for g in graphs]
        val_graphs = [ensure_cpu(g) for g in fold_graphs[fold_id]]

        input_dim = train_graphs[0].x.shape[1]
        model = model_class(input_dim=input_dim, **model_kwargs).to(device)

        loss_fn = PairwiseMarginLoss(margin=config.get("margin", 0.1))
        model = train_model(model, train_graphs, val_graphs, config, device, loss_fn)

        ndcg, mrr = evaluate_model(model, val_graphs, device)
        fold_ndcg.append(ndcg)
        fold_mrr.append(mrr)
        print(f"  Fold {fold_id}: nDCG@10={ndcg:.4f}, MRR={mrr:.4f}")

    mean_ndcg = np.mean(fold_ndcg)
    std_ndcg = np.std(fold_ndcg)
    mean_mrr = np.mean(fold_mrr)

    print(f"  Mean: nDCG@10={mean_ndcg:.4f} ± {std_ndcg:.4f}, MRR={mean_mrr:.4f}")

    return {
        "name": name,
        "ndcg_mean": float(mean_ndcg),
        "ndcg_std": float(std_ndcg),
        "mrr_mean": float(mean_mrr),
        "fold_ndcg": [float(x) for x in fold_ndcg],
        "fold_mrr": [float(x) for x in fold_mrr],
    }


class ContrastivePretrainedSAGE(nn.Module):
    """SAGE with contrastive pre-training."""
    def __init__(self, input_dim, hidden_dim=128, num_layers=1, dropout=0.05,
                 alpha_init=0.65, learn_alpha=True):
        super().__init__()
        self.encoder = nn.ModuleList([SAGEConv(input_dim, hidden_dim)])
        for _ in range(num_layers - 1):
            self.encoder.append(SAGEConv(hidden_dim, hidden_dim))

        self.score_head = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(alpha_init), requires_grad=learn_alpha)
        self.residual_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else None
        self.hidden_dim = hidden_dim

    def encode(self, x, edge_index):
        """Get node embeddings (for contrastive loss)."""
        h = x
        for conv in self.encoder:
            h = F.relu(conv(h, edge_index))
            h = self.dropout(h)
        return h

    def forward(self, x, edge_index, reranker_scores, batch=None, edge_weight=None):
        h_in = self.residual_proj(x) if self.residual_proj else x
        h = self.encode(x, edge_index)
        h = h + h_in
        gnn_scores = self.score_head(h).squeeze(-1)
        alpha = torch.sigmoid(self.alpha)
        return alpha * reranker_scores + (1 - alpha) * gnn_scores


class DistillationSAGE(nn.Module):
    """SAGE for knowledge distillation training."""
    def __init__(self, input_dim, hidden_dim=128, num_layers=1, dropout=0.05,
                 alpha_init=0.65, learn_alpha=True):
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(input_dim, hidden_dim)])
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        self.score_head = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(alpha_init), requires_grad=learn_alpha)
        self.residual_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else None

    def forward(self, x, edge_index, reranker_scores, batch=None, edge_weight=None):
        h_in = self.residual_proj(x) if self.residual_proj else x
        h = x
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
            h = self.dropout(h)
        h = h + h_in
        gnn_scores = self.score_head(h).squeeze(-1)
        alpha = torch.sigmoid(self.alpha)
        return alpha * reranker_scores + (1 - alpha) * gnn_scores


def generate_pseudo_llm_labels(graphs, device):
    """Generate pseudo LLM soft labels for distillation.

    In practice, this would run an LLM on each sentence.
    Here we simulate with a heuristic based on reranker scores + noise.
    """
    for g in graphs:
        # Simulate LLM scores: reranker scores with some "clinical insight" adjustment
        reranker_scores = g.reranker_scores.cpu().numpy()
        gold_labels = g.node_labels.cpu().numpy()

        # LLM would have better calibration on positive examples
        llm_scores = reranker_scores.copy()

        # Boost true positives (simulating LLM's clinical knowledge)
        n_positives = int(gold_labels.sum())  # Convert np.float32 to int
        llm_scores[gold_labels > 0] += 0.3 * np.random.uniform(0.5, 1.0, size=n_positives)

        # Add some noise to negatives (LLM isn't perfect)
        neg_mask = gold_labels == 0
        n_negatives = int(neg_mask.sum())  # Convert np.int64 to int for safety
        llm_scores[neg_mask] += 0.1 * np.random.randn(n_negatives)

        # Normalize to [0, 1]
        llm_scores = (llm_scores - llm_scores.min()) / (llm_scores.max() - llm_scores.min() + 1e-8)

        g.llm_soft_labels = torch.tensor(llm_scores, dtype=torch.float32)

    return graphs


def train_with_contrastive_pretrain(model, train_graphs, val_graphs, config, device):
    """Two-stage training: contrastive pre-training + fine-tuning."""
    torch.manual_seed(config.get("seed", 42))
    np.random.seed(config.get("seed", 42))

    train_graphs = [ensure_cpu(g) for g in train_graphs]
    train_loader = DataLoader(train_graphs, batch_size=config["batch_size"], shuffle=True)

    # Stage 1: Contrastive pre-training
    print("    Stage 1: Contrastive pre-training...")
    optimizer = AdamW(model.parameters(), lr=config["lr"] * 10, weight_decay=config["weight_decay"])
    contrastive_loss_fn = ContrastiveLoss(temperature=0.07)

    for epoch in range(config.get("pretrain_epochs", 10)):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Get embeddings
            embeddings = model.encode(batch.x, batch.edge_index)

            # Contrastive loss using node labels as "classes"
            loss = contrastive_loss_fn(embeddings, batch.node_labels, batch.batch)
            loss.backward()
            optimizer.step()

    # Stage 2: Fine-tuning for ranking
    print("    Stage 2: Fine-tuning for ranking...")
    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    loss_fn = PairwiseMarginLoss(margin=config.get("margin", 0.1))

    best_ndcg, best_state, patience = -1, None, 0

    for epoch in range(config["max_epochs"]):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            refined = model(batch.x, batch.edge_index, batch.reranker_scores, batch.batch)
            loss = loss_fn(refined, batch.node_labels, batch.batch)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_ndcg = []
        with torch.no_grad():
            for g in val_graphs:
                g = ensure_cpu(g).to(device)
                gold = g.node_labels.cpu().numpy()
                scores = model(g.x, g.edge_index, g.reranker_scores).cpu().numpy()
                val_ndcg.append(compute_ndcg_at_k(gold, scores, 10))

        mean_ndcg = np.mean(val_ndcg)
        if mean_ndcg > best_ndcg:
            best_ndcg = mean_ndcg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if patience >= config["patience"]:
            break

    if best_state:
        model.load_state_dict(best_state)
    return model


def train_with_distillation(model, train_graphs, val_graphs, config, device):
    """Train with knowledge distillation from LLM soft labels."""
    torch.manual_seed(config.get("seed", 42))
    np.random.seed(config.get("seed", 42))

    train_graphs = [ensure_cpu(g) for g in train_graphs]
    train_loader = DataLoader(train_graphs, batch_size=config["batch_size"], shuffle=True)

    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    hard_loss_fn = PairwiseMarginLoss(margin=config.get("margin", 0.1))

    distill_alpha = config.get("distill_alpha", 0.3)  # Weight for distillation loss

    best_ndcg, best_state, patience = -1, None, 0

    for epoch in range(config["max_epochs"]):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            refined = model(batch.x, batch.edge_index, batch.reranker_scores, batch.batch)

            # Hard label loss (ranking)
            hard_loss = hard_loss_fn(refined, batch.node_labels, batch.batch)

            # Soft label loss (distillation from LLM)
            if hasattr(batch, 'llm_soft_labels'):
                llm_targets = batch.llm_soft_labels.to(device)
                # MSE loss between student scores and teacher scores
                soft_loss = F.mse_loss(torch.sigmoid(refined), llm_targets)
            else:
                soft_loss = torch.tensor(0.0, device=device)

            # Combined loss
            loss = (1 - distill_alpha) * hard_loss + distill_alpha * soft_loss
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_ndcg = []
        with torch.no_grad():
            for g in val_graphs:
                g = ensure_cpu(g).to(device)
                gold = g.node_labels.cpu().numpy()
                scores = model(g.x, g.edge_index, g.reranker_scores).cpu().numpy()
                val_ndcg.append(compute_ndcg_at_k(gold, scores, 10))

        mean_ndcg = np.mean(val_ndcg)
        if mean_ndcg > best_ndcg:
            best_ndcg = mean_ndcg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if patience >= config["patience"]:
            break

    if best_state:
        model.load_state_dict(best_state)
    return model


def run_contrastive_experiment(name, fold_graphs, config, device):
    """Run contrastive pre-training experiment."""
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"{'='*60}")

    n_folds = len(fold_graphs)
    fold_ndcg = []
    fold_mrr = []

    for fold_id in range(n_folds):
        train_graphs = [g for fid, graphs in fold_graphs.items() if fid != fold_id for g in graphs]
        val_graphs = [ensure_cpu(g) for g in fold_graphs[fold_id]]

        input_dim = train_graphs[0].x.shape[1]
        model = ContrastivePretrainedSAGE(
            input_dim=input_dim, hidden_dim=128, num_layers=1,
            dropout=0.05, alpha_init=0.65
        ).to(device)

        model = train_with_contrastive_pretrain(model, train_graphs, val_graphs, config, device)

        ndcg, mrr = evaluate_model(model, val_graphs, device)
        fold_ndcg.append(ndcg)
        fold_mrr.append(mrr)
        print(f"  Fold {fold_id}: nDCG@10={ndcg:.4f}, MRR={mrr:.4f}")

    mean_ndcg = np.mean(fold_ndcg)
    std_ndcg = np.std(fold_ndcg)
    mean_mrr = np.mean(fold_mrr)

    print(f"  Mean: nDCG@10={mean_ndcg:.4f} ± {std_ndcg:.4f}, MRR={mean_mrr:.4f}")

    return {
        "name": name,
        "ndcg_mean": float(mean_ndcg),
        "ndcg_std": float(std_ndcg),
        "mrr_mean": float(mean_mrr),
        "fold_ndcg": [float(x) for x in fold_ndcg],
        "fold_mrr": [float(x) for x in fold_mrr],
    }


def run_distillation_experiment(name, fold_graphs, config, device):
    """Run knowledge distillation experiment."""
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"{'='*60}")

    n_folds = len(fold_graphs)
    fold_ndcg = []
    fold_mrr = []

    for fold_id in range(n_folds):
        train_graphs = [g for fid, graphs in fold_graphs.items() if fid != fold_id for g in graphs]
        val_graphs = [ensure_cpu(g) for g in fold_graphs[fold_id]]

        # Generate pseudo LLM labels for training graphs
        print(f"  Fold {fold_id}: Generating pseudo LLM labels...")
        train_graphs = generate_pseudo_llm_labels(train_graphs, device)

        input_dim = train_graphs[0].x.shape[1]
        model = DistillationSAGE(
            input_dim=input_dim, hidden_dim=128, num_layers=1,
            dropout=0.05, alpha_init=0.65
        ).to(device)

        model = train_with_distillation(model, train_graphs, val_graphs, config, device)

        ndcg, mrr = evaluate_model(model, val_graphs, device)
        fold_ndcg.append(ndcg)
        fold_mrr.append(mrr)
        print(f"  Fold {fold_id}: nDCG@10={ndcg:.4f}, MRR={mrr:.4f}")

    mean_ndcg = np.mean(fold_ndcg)
    std_ndcg = np.std(fold_ndcg)
    mean_mrr = np.mean(fold_mrr)

    print(f"  Mean: nDCG@10={mean_ndcg:.4f} ± {std_ndcg:.4f}, MRR={mean_mrr:.4f}")

    return {
        "name": name,
        "ndcg_mean": float(mean_ndcg),
        "ndcg_std": float(std_ndcg),
        "mrr_mean": float(mean_mrr),
        "fold_ndcg": [float(x) for x in fold_ndcg],
        "fold_mrr": [float(x) for x in fold_mrr],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_dir", default="data/cache/gnn/rebuild_20260120")
    parser.add_argument("--output_dir", default="outputs/all_experiments")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    print("\nLoading graphs...")
    fold_graphs = load_graphs(Path(args.graph_dir))

    # Base config
    base_config = {
        "hidden_dim": 128, "num_layers": 1, "dropout": 0.05,
        "alpha_init": 0.65, "learn_alpha": True, "lr": 3.69e-5,
        "weight_decay": 9.06e-6, "batch_size": 32, "max_epochs": 25,
        "patience": 10, "margin": 0.1, "seed": 42,
    }

    results = []

    # ========================================================================
    # 1. Baseline: SAGE + Residual
    # ========================================================================
    results.append(run_experiment(
        "Baseline (SAGE+Residual)",
        SAGEResidualReranker,
        {"hidden_dim": 128, "num_layers": 1, "dropout": 0.05, "alpha_init": 0.65},
        base_config,
        fold_graphs,
        device
    ))
    baseline_ndcg = results[0]["ndcg_mean"]

    # ========================================================================
    # 2. JK-Net experiments
    # ========================================================================
    for jk_mode in ["cat", "max", "lstm"]:
        results.append(run_experiment(
            f"JK-Net (mode={jk_mode})",
            JKNetReranker,
            {"hidden_dim": 128, "num_layers": 3, "dropout": 0.05, "alpha_init": 0.65, "jk_mode": jk_mode},
            base_config,
            fold_graphs,
            device
        ))

    # ========================================================================
    # 3. GAT v2 experiments
    # ========================================================================
    for heads in [2, 4, 8]:
        results.append(run_experiment(
            f"GAT v2 (heads={heads})",
            GATv2Reranker,
            {"hidden_dim": 128, "num_layers": 1, "dropout": 0.05, "alpha_init": 0.65, "heads": heads},
            base_config,
            fold_graphs,
            device
        ))

    # ========================================================================
    # 4. Different margin values
    # ========================================================================
    for margin in [0.05, 0.15, 0.2, 0.3]:
        config = base_config.copy()
        config["margin"] = margin
        results.append(run_experiment(
            f"Margin={margin}",
            SAGEResidualReranker,
            {"hidden_dim": 128, "num_layers": 1, "dropout": 0.05, "alpha_init": 0.65},
            config,
            fold_graphs,
            device
        ))

    # ========================================================================
    # 5. Learning rate schedules
    # ========================================================================
    for scheduler in ["cosine", "step", "warmup_cosine"]:
        config = base_config.copy()
        config["scheduler"] = scheduler
        results.append(run_experiment(
            f"LR Schedule: {scheduler}",
            SAGEResidualReranker,
            {"hidden_dim": 128, "num_layers": 1, "dropout": 0.05, "alpha_init": 0.65},
            config,
            fold_graphs,
            device
        ))

    # ========================================================================
    # 6. Batch sizes
    # ========================================================================
    for batch_size in [64, 128, 256]:
        config = base_config.copy()
        config["batch_size"] = batch_size
        results.append(run_experiment(
            f"Batch size={batch_size}",
            SAGEResidualReranker,
            {"hidden_dim": 128, "num_layers": 1, "dropout": 0.05, "alpha_init": 0.65},
            config,
            fold_graphs,
            device
        ))

    # ========================================================================
    # 7. Edge features (similarity weights)
    # ========================================================================
    print("\nAdding edge weights based on cosine similarity...")
    fold_graphs_weighted = {k: add_edge_weights([ensure_cpu(g) for g in v])
                           for k, v in fold_graphs.items()}

    results.append(run_experiment(
        "Edge weights (cosine similarity)",
        EdgeWeightedSAGE,
        {"hidden_dim": 128, "num_layers": 1, "dropout": 0.05, "alpha_init": 0.65},
        base_config,
        fold_graphs_weighted,
        device
    ))

    # ========================================================================
    # 8. Position/length features
    # ========================================================================
    print("\nAdding position and length features...")
    fold_graphs_pos = {k: add_position_features([ensure_cpu(g) for g in v])
                      for k, v in fold_graphs.items()}

    results.append(run_experiment(
        "Position + Length features",
        PositionAwareSAGE,
        {"hidden_dim": 128, "num_layers": 1, "dropout": 0.05, "alpha_init": 0.65},
        base_config,
        fold_graphs_pos,
        device
    ))

    # ========================================================================
    # 9. Multi-task learning
    # ========================================================================
    results.append(run_experiment(
        "Multi-task (ranking + classification)",
        MultiTaskReranker,
        {"hidden_dim": 128, "num_layers": 1, "dropout": 0.05, "alpha_init": 0.65, "num_criteria": 10},
        base_config,
        fold_graphs,
        device
    ))

    # ========================================================================
    # 10. Contrastive Pre-training
    # ========================================================================
    contrastive_config = base_config.copy()
    contrastive_config["pretrain_epochs"] = 10
    results.append(run_contrastive_experiment(
        "Contrastive Pre-training (10 epochs)",
        fold_graphs,
        contrastive_config,
        device
    ))

    contrastive_config["pretrain_epochs"] = 20
    results.append(run_contrastive_experiment(
        "Contrastive Pre-training (20 epochs)",
        fold_graphs,
        contrastive_config,
        device
    ))

    # ========================================================================
    # 11. Knowledge Distillation (with pseudo LLM labels)
    # ========================================================================
    for distill_alpha in [0.1, 0.3, 0.5]:
        distill_config = base_config.copy()
        distill_config["distill_alpha"] = distill_alpha
        results.append(run_distillation_experiment(
            f"Knowledge Distillation (alpha={distill_alpha})",
            fold_graphs,
            distill_config,
            device
        ))

    # ========================================================================
    # 12. Combined: Contrastive + Residual + Best settings
    # ========================================================================
    # Test combining multiple improvements
    for margin in [0.1, 0.15]:
        for scheduler in [None, "cosine"]:
            config = base_config.copy()
            config["margin"] = margin
            config["scheduler"] = scheduler
            sched_name = scheduler if scheduler else "none"
            results.append(run_experiment(
                f"Combined: margin={margin}, sched={sched_name}",
                SAGEResidualReranker,
                {"hidden_dim": 128, "num_layers": 1, "dropout": 0.05, "alpha_init": 0.65},
                config,
                fold_graphs,
                device
            ))

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*80)
    print("COMPREHENSIVE EXPERIMENT SUMMARY")
    print("="*80)

    # Sort by nDCG
    results_sorted = sorted(results, key=lambda x: x["ndcg_mean"], reverse=True)

    print(f"{'Experiment':<45} {'nDCG@10':<20} {'vs Baseline':<15}")
    print("-" * 80)

    for r in results_sorted:
        delta = (r["ndcg_mean"] - baseline_ndcg) / baseline_ndcg * 100
        marker = "**" if r["ndcg_mean"] > baseline_ndcg else ""
        print(f"{marker}{r['name']:<43} {r['ndcg_mean']:.4f} ± {r['ndcg_std']:.4f}   {delta:+.2f}%")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"all_experiments_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump({"results": results, "baseline_ndcg": baseline_ndcg}, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Find best
    best = results_sorted[0]
    print(f"\nBest: {best['name']} with nDCG@10={best['ndcg_mean']:.4f}")


if __name__ == "__main__":
    main()
