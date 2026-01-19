"""Clinical Model Inference Module.

This module loads trained P3 (graph reranker) and P4 (NE gate) GNN models
and runs inference for clinical deployment.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


class ClinicalModelInference:
    """Load and run inference with trained P3/P4 models for clinical deployment.

    This class handles:
    1. Loading model checkpoints from disk
    2. Running batched inference on graphs
    3. Extracting predictions in clinical-ready format
    """

    def __init__(
        self,
        p3_model_path: Optional[str] = None,
        p4_model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 32,
    ):
        """Initialize clinical model inference.

        Args:
            p3_model_path: Path to P3 graph reranker checkpoint
            p4_model_path: Path to P4 NE gate checkpoint
            device: Device to run inference on
            batch_size: Batch size for inference
        """
        self.device = torch.device(device)
        self.batch_size = batch_size

        # Load P3 model if provided
        self.p3_model = None
        if p3_model_path is not None:
            self.p3_model = self._load_p3_model(p3_model_path)
            logger.info(f"Loaded P3 model from {p3_model_path}")

        # Load P4 model if provided
        self.p4_model = None
        if p4_model_path is not None:
            self.p4_model = self._load_p4_model(p4_model_path)
            logger.info(f"Loaded P4 model from {p4_model_path}")

    def _load_p3_model(self, model_path: str):
        """Load P3 graph reranker model from checkpoint."""
        from final_sc_review.gnn.models.p3_graph_reranker import GraphRerankerGNN

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Extract model config from checkpoint
        if 'config' in checkpoint:
            config = checkpoint['config']
            input_dim = config.get('input_dim', 1032)
            hidden_dim = config.get('hidden_dim', 256)
            num_layers = config.get('num_layers', 2)
            dropout = config.get('dropout', 0.2)
        else:
            # Use defaults
            input_dim = 1032
            hidden_dim = 256
            num_layers = 2
            dropout = 0.2

        # Create model
        model = GraphRerankerGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(self.device)
        model.eval()

        return model

    def _load_p4_model(self, model_path: str):
        """Load P4 NE gate model from checkpoint."""
        # Import the actual P4 model used (CriterionAwareNEGNN)
        import sys
        from pathlib import Path as PathlibPath

        # Add scripts to path to access the model definition
        scripts_path = PathlibPath(__file__).parent.parent.parent.parent / "scripts" / "gnn"
        sys.path.insert(0, str(scripts_path))

        try:
            from train_eval_hetero_graph import CriterionAwareNEGNN
        except ImportError:
            logger.error("Could not import CriterionAwareNEGNN. Ensure scripts/gnn/train_eval_hetero_graph.py is available.")
            raise

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Extract model config from checkpoint
        if 'config' in checkpoint:
            config = checkpoint['config']
            input_dim = config.get('input_dim', 1032)
            criterion_dim = config.get('criterion_dim', 64)
            num_criteria = config.get('num_criteria', 10)
            hidden_dim = config.get('hidden_dim', 256)
            num_layers = config.get('num_layers', 3)
            num_heads = config.get('num_heads', 4)
            dropout = config.get('dropout', 0.3)
        else:
            # Use defaults from the training script
            input_dim = 1032
            criterion_dim = 64
            num_criteria = 10
            hidden_dim = 256
            num_layers = 3
            num_heads = 4
            dropout = 0.3

        # Create model
        model = CriterionAwareNEGNN(
            input_dim=input_dim,
            criterion_dim=criterion_dim,
            num_criteria=num_criteria,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(self.device)
        model.eval()

        return model

    def extract_criterion_id(self, graph: Data) -> int:
        """Extract criterion ID from graph."""
        # Try criterion_id first
        if hasattr(graph, 'criterion_id'):
            crit_id = graph.criterion_id
            # If it's already an int, return it
            if isinstance(crit_id, int):
                return crit_id
            # If it's a string, parse it
            if isinstance(crit_id, str) and crit_id.startswith("A."):
                try:
                    return int(crit_id.split(".")[1]) - 1
                except:
                    pass

        # Try criterion attribute
        if hasattr(graph, 'criterion'):
            crit = graph.criterion
            if isinstance(crit, int):
                return crit
            if isinstance(crit, str) and crit.startswith("A."):
                try:
                    return int(crit.split(".")[1]) - 1
                except:
                    pass

        # Default to 0 if cannot determine
        return 0

    def extract_original_scores(self, graph: Data) -> torch.Tensor:
        """Extract original reranker scores from graph node features."""
        # Node features: [embedding (1024) + score (1) + rank_percentile (1) + score_gaps (2) + stats (4)]
        # Score is at index 1024
        if graph.x.shape[1] >= 1025:
            return graph.x[:, 1024]  # Extract score column
        else:
            # Fallback: use zeros
            return torch.zeros(graph.x.shape[0], device=graph.x.device)

    @torch.no_grad()
    def predict_p3_batch(self, graphs: List[Data]) -> List[np.ndarray]:
        """Run P3 inference to get refined scores.

        Args:
            graphs: List of PyG Data objects

        Returns:
            List of refined score arrays (one per graph)
        """
        if self.p3_model is None:
            logger.warning("P3 model not loaded, returning original scores")
            # Return original scores as fallback
            return [self.extract_original_scores(g).cpu().numpy() for g in graphs]

        loader = DataLoader(graphs, batch_size=self.batch_size, shuffle=False)

        all_refined_scores = []

        for batch in loader:
            batch = batch.to(self.device)

            # Extract original scores from batch
            original_scores = self.extract_original_scores(batch)

            # Run P3 inference
            refined_scores = self.p3_model(
                x=batch.x,
                edge_index=batch.edge_index,
                original_scores=original_scores,
                batch=batch.batch,
            )

            # Split back into individual graphs
            batch_sizes = batch.batch.bincount().tolist()
            start = 0
            for size in batch_sizes:
                end = start + size
                all_refined_scores.append(refined_scores[start:end].cpu().numpy())
                start = end

        return all_refined_scores

    @torch.no_grad()
    def predict_p4_batch(self, graphs: List[Data]) -> np.ndarray:
        """Run P4 inference to get has_evidence probabilities.

        Args:
            graphs: List of PyG Data objects

        Returns:
            Array of probabilities [n_graphs]
        """
        if self.p4_model is None:
            logger.warning("P4 model not loaded, returning default probabilities")
            # Return 0.5 as fallback
            return np.full(len(graphs), 0.5)

        loader = DataLoader(graphs, batch_size=self.batch_size, shuffle=False)

        all_probs = []
        graph_offset = 0  # Track which graphs we've processed

        for batch in loader:
            batch = batch.to(self.device)

            # Extract criterion IDs for each graph in current batch
            batch_sizes = batch.batch.bincount().tolist()
            n_graphs_in_batch = len(batch_sizes)

            criterion_ids = []
            for graph_idx in range(n_graphs_in_batch):
                original_graph = graphs[graph_offset + graph_idx]
                criterion_ids.append(self.extract_criterion_id(original_graph))

            criterion_ids = torch.tensor(criterion_ids, dtype=torch.long, device=self.device)

            # Run P4 inference
            logits = self.p4_model(
                x=batch.x,
                edge_index=batch.edge_index,
                batch=batch.batch,
                criterion_ids=criterion_ids,
            )

            # Convert to probabilities
            probs = torch.sigmoid(logits).squeeze(-1)
            all_probs.append(probs.cpu().numpy())

            graph_offset += n_graphs_in_batch

        return np.concatenate(all_probs)

    def augment_graphs_with_predictions(
        self,
        graphs: List[Data],
        run_p3: bool = True,
        run_p4: bool = True,
    ) -> List[Data]:
        """Augment graphs with P3/P4 predictions.

        This adds `p3_scores` and `p4_prob` attributes to each graph.

        Args:
            graphs: List of graphs to augment
            run_p3: Whether to run P3 inference
            run_p4: Whether to run P4 inference

        Returns:
            Augmented graphs (modifies in place)
        """
        logger.info(f"Running inference on {len(graphs)} graphs...")

        # Run P3 inference
        if run_p3:
            logger.info("Running P3 (graph reranker) inference...")
            p3_scores = self.predict_p3_batch(graphs)
            for graph, scores in zip(graphs, p3_scores):
                graph.p3_scores = torch.from_numpy(scores).float()

        # Run P4 inference
        if run_p4:
            logger.info("Running P4 (NE gate) inference...")
            p4_probs = self.predict_p4_batch(graphs)
            for graph, prob in zip(graphs, p4_probs):
                graph.p4_prob = float(prob)

        logger.info("Inference complete!")

        return graphs


def load_and_augment_fold(
    graph_dir: Path,
    fold_id: int,
    p3_model_path: Optional[str] = None,
    p4_model_path: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> List[Data]:
    """Load a fold and augment with model predictions.

    Args:
        graph_dir: Directory containing fold files
        fold_id: Fold index (0-4)
        p3_model_path: Path to P3 model checkpoint
        p4_model_path: Path to P4 model checkpoint
        device: Device for inference

    Returns:
        List of augmented graphs
    """
    # Load fold
    fold_file = graph_dir / f"fold_{fold_id}.pt"
    data = torch.load(fold_file, weights_only=False)
    graphs = data['graphs'] if isinstance(data, dict) else data

    logger.info(f"Loaded {len(graphs)} graphs from fold {fold_id}")

    # Create inference module
    inference = ClinicalModelInference(
        p3_model_path=p3_model_path,
        p4_model_path=p4_model_path,
        device=device,
    )

    # Augment with predictions
    graphs = inference.augment_graphs_with_predictions(graphs)

    return graphs
