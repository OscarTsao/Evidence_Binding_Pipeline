#!/usr/bin/env python3
"""Real latency measurement for efficiency metrics.

Measures actual inference time for each pipeline component:
1. Retrieval (NV-Embed-v2 encoding + similarity search)
2. Reranking (Jina-Reranker-v3)
3. GNN modules (P2, P3, P4)
4. End-to-end pipeline

Usage:
    python scripts/analysis/measure_latency.py \
        --output outputs/efficiency/ \
        --n_samples 100 \
        --warmup 5
"""

import argparse
import gc
import json
import logging
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def get_hardware_info() -> Dict:
    """Get hardware information for reproducibility."""
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
    }

    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    except ImportError:
        info["torch_available"] = False

    try:
        import psutil
        info["cpu_count"] = psutil.cpu_count()
        info["memory_gb"] = psutil.virtual_memory().total / 1e9
    except ImportError:
        pass

    return info


def time_function(func, *args, n_runs: int = 10, warmup: int = 2, **kwargs) -> Dict:
    """Time a function with warmup and multiple runs.

    Args:
        func: Function to time
        args: Positional arguments
        n_runs: Number of timed runs
        warmup: Number of warmup runs (not timed)
        kwargs: Keyword arguments

    Returns:
        Dictionary with timing statistics in milliseconds
    """
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    # Warmup runs
    for _ in range(warmup):
        try:
            func(*args, **kwargs)
        except Exception:
            pass

    # Timed runs
    times = []
    for _ in range(n_runs):
        gc.collect()

        if has_cuda:
            import torch
            torch.cuda.synchronize()

        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Function failed: {e}")
            continue

        if has_cuda:
            import torch
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    if not times:
        return {"error": "All runs failed"}

    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "median_ms": float(np.median(times)),
        "n_runs": len(times),
    }


def measure_retriever_latency(
    n_samples: int = 100,
    warmup: int = 5,
) -> Dict:
    """Measure retriever (NV-Embed-v2) latency.

    Returns timing for:
    - Query encoding
    - Corpus encoding (per sentence)
    - Similarity search
    """
    results = {}

    try:
        from final_sc_review.retriever.zoo import NVEmbedV2Retriever
        logger.info("Measuring NV-Embed-v2 retriever latency...")

        # Create dummy data
        dummy_query = "Depressed mood most of the day, nearly every day"
        dummy_sentences = [f"This is test sentence {i} with some content." for i in range(20)]

        # Time query encoding
        retriever = NVEmbedV2Retriever()

        def encode_query():
            return retriever.encode_queries([dummy_query])

        results["query_encoding"] = time_function(encode_query, n_runs=n_samples, warmup=warmup)

        # Time sentence encoding (batch of 20)
        def encode_sentences():
            return retriever.encode_sentences(dummy_sentences)

        results["sentence_encoding_batch20"] = time_function(encode_sentences, n_runs=min(n_samples, 20), warmup=warmup)

        # Per-sentence time
        if "mean_ms" in results["sentence_encoding_batch20"]:
            results["sentence_encoding_per_sentence"] = {
                "mean_ms": results["sentence_encoding_batch20"]["mean_ms"] / 20
            }

        logger.info(f"  Query encoding: {results['query_encoding'].get('mean_ms', 'N/A'):.1f}ms")
        logger.info(f"  Sentence encoding (20): {results['sentence_encoding_batch20'].get('mean_ms', 'N/A'):.1f}ms")

    except Exception as e:
        logger.warning(f"Could not measure retriever: {e}")
        results["error"] = str(e)

    return results


def measure_reranker_latency(
    n_samples: int = 100,
    warmup: int = 5,
) -> Dict:
    """Measure reranker (Jina-Reranker-v3) latency."""
    results = {}

    try:
        from final_sc_review.reranker.zoo import JinaRerankerV3
        logger.info("Measuring Jina-Reranker-v3 latency...")

        # Create dummy data
        dummy_query = "Depressed mood most of the day, nearly every day"
        dummy_pairs = [(dummy_query, f"Test sentence {i}") for i in range(24)]

        reranker = JinaRerankerV3()

        # Time reranking
        def rerank():
            return reranker.rerank(dummy_query, [p[1] for p in dummy_pairs], top_k=10)

        results["rerank_24_candidates"] = time_function(rerank, n_runs=n_samples, warmup=warmup)

        if "mean_ms" in results["rerank_24_candidates"]:
            results["rerank_per_candidate"] = {
                "mean_ms": results["rerank_24_candidates"]["mean_ms"] / 24
            }

        logger.info(f"  Rerank 24 candidates: {results['rerank_24_candidates'].get('mean_ms', 'N/A'):.1f}ms")

    except Exception as e:
        logger.warning(f"Could not measure reranker: {e}")
        results["error"] = str(e)

    return results


def measure_gnn_latency(
    n_samples: int = 100,
    warmup: int = 5,
) -> Dict:
    """Measure GNN module latency."""
    results = {}

    try:
        import torch
        from torch_geometric.data import Data

        # Create dummy graph
        n_nodes = 20
        x = torch.randn(n_nodes, 4098)  # NV-Embed-v2 dim + features
        edge_index = torch.randint(0, n_nodes, (2, n_nodes * 2))
        scores = torch.randn(n_nodes)

        dummy_graph = Data(x=x, edge_index=edge_index, scores=scores)

        # Try to load P3 model
        try:
            from final_sc_review.gnn.models.p3_graph_reranker import GraphRerankerGNN

            model = GraphRerankerGNN(input_dim=4098, hidden_dim=128, num_layers=2)
            model.eval()

            if torch.cuda.is_available():
                model = model.cuda()
                dummy_graph = dummy_graph.cuda()

            def p3_forward():
                with torch.no_grad():
                    return model(dummy_graph)

            results["p3_graph_reranker"] = time_function(p3_forward, n_runs=n_samples, warmup=warmup)
            logger.info(f"  P3 Graph Reranker: {results['p3_graph_reranker'].get('mean_ms', 'N/A'):.1f}ms")

        except Exception as e:
            logger.warning(f"Could not measure P3: {e}")
            results["p3_error"] = str(e)

    except ImportError as e:
        logger.warning(f"GNN dependencies not available: {e}")
        results["error"] = str(e)

    return results


def measure_end_to_end_latency(
    n_samples: int = 20,
    warmup: int = 2,
) -> Dict:
    """Measure end-to-end pipeline latency."""
    results = {}

    try:
        from final_sc_review.pipeline.zoo_pipeline import ZooPipeline
        from final_sc_review.data.io import load_sentence_corpus

        logger.info("Measuring end-to-end pipeline latency...")

        # Load sentence corpus for realistic test
        corpus_path = Path("data/groundtruth/sentence_corpus.jsonl")
        if not corpus_path.exists():
            logger.warning("Sentence corpus not found, skipping end-to-end test")
            return {"error": "Sentence corpus not found"}

        sentences = load_sentence_corpus(corpus_path)

        # Get a sample post
        from collections import defaultdict
        post_to_sents = defaultdict(list)
        for s in sentences:
            post_to_sents[s.post_id].append(s)

        # Pick a post with reasonable number of sentences
        sample_post_id = None
        for pid, sents in post_to_sents.items():
            if 15 <= len(sents) <= 30:
                sample_post_id = pid
                break

        if not sample_post_id:
            sample_post_id = list(post_to_sents.keys())[0]

        sample_sentences = post_to_sents[sample_post_id]
        logger.info(f"  Using post {sample_post_id} with {len(sample_sentences)} sentences")

        # Initialize pipeline (this will be slow due to model loading)
        pipeline = ZooPipeline(
            retriever_name="nv-embed-v2",
            reranker_name="jina-reranker-v3",
        )

        dummy_query = "Depressed mood most of the day, nearly every day"

        # Time full pipeline
        def run_pipeline():
            return pipeline.retrieve(dummy_query, sample_post_id)

        results["full_pipeline"] = time_function(run_pipeline, n_runs=n_samples, warmup=warmup)
        logger.info(f"  Full pipeline: {results['full_pipeline'].get('mean_ms', 'N/A'):.1f}ms")

    except Exception as e:
        logger.warning(f"Could not measure end-to-end: {e}")
        results["error"] = str(e)

    return results


def generate_efficiency_report(
    results: Dict,
    hardware: Dict,
    output_dir: Path,
) -> None:
    """Generate efficiency metrics report."""
    report = f"""# Efficiency Metrics Report

Generated: {datetime.now().isoformat()}

---

## Hardware Configuration

| Component | Value |
|-----------|-------|
| Platform | {hardware.get('platform', 'N/A')} |
| Python | {hardware.get('python_version', 'N/A')} |
| PyTorch | {hardware.get('torch_version', 'N/A')} |
| CUDA | {hardware.get('cuda_version', 'N/A') if hardware.get('cuda_available') else 'Not available'} |
| GPU | {hardware.get('gpu_name', 'N/A')} |
| GPU Memory | {hardware.get('gpu_memory_gb', 'N/A'):.1f} GB |
| CPU Cores | {hardware.get('cpu_count', 'N/A')} |
| RAM | {hardware.get('memory_gb', 'N/A'):.1f} GB |

---

## Latency Measurements

### Retriever (NV-Embed-v2)

"""
    if "retriever" in results:
        r = results["retriever"]
        if "query_encoding" in r:
            qe = r["query_encoding"]
            report += f"- **Query encoding**: {qe.get('mean_ms', 'N/A'):.1f} ± {qe.get('std_ms', 0):.1f} ms\n"
        if "sentence_encoding_batch20" in r:
            se = r["sentence_encoding_batch20"]
            report += f"- **Sentence encoding (batch=20)**: {se.get('mean_ms', 'N/A'):.1f} ± {se.get('std_ms', 0):.1f} ms\n"
        if "sentence_encoding_per_sentence" in r:
            sps = r["sentence_encoding_per_sentence"]
            report += f"- **Per-sentence encoding**: {sps.get('mean_ms', 'N/A'):.2f} ms\n"

    report += "\n### Reranker (Jina-Reranker-v3)\n\n"
    if "reranker" in results:
        r = results["reranker"]
        if "rerank_24_candidates" in r:
            rr = r["rerank_24_candidates"]
            report += f"- **Rerank 24 candidates**: {rr.get('mean_ms', 'N/A'):.1f} ± {rr.get('std_ms', 0):.1f} ms\n"
        if "rerank_per_candidate" in r:
            rpc = r["rerank_per_candidate"]
            report += f"- **Per-candidate**: {rpc.get('mean_ms', 'N/A'):.2f} ms\n"

    report += "\n### GNN Modules\n\n"
    if "gnn" in results:
        r = results["gnn"]
        if "p3_graph_reranker" in r:
            p3 = r["p3_graph_reranker"]
            report += f"- **P3 Graph Reranker**: {p3.get('mean_ms', 'N/A'):.1f} ± {p3.get('std_ms', 0):.1f} ms\n"

    report += "\n### End-to-End Pipeline\n\n"
    if "end_to_end" in results:
        r = results["end_to_end"]
        if "full_pipeline" in r:
            fp = r["full_pipeline"]
            report += f"- **Full pipeline**: {fp.get('mean_ms', 'N/A'):.1f} ± {fp.get('std_ms', 0):.1f} ms\n"
            if "mean_ms" in fp:
                qps = 1000 / fp["mean_ms"]
                report += f"- **Throughput**: {qps:.1f} queries/second\n"

    report += """
---

## Summary

| Stage | Latency (ms) |
|-------|-------------|
"""

    # Aggregate summary
    if "retriever" in results and "query_encoding" in results["retriever"]:
        report += f"| Retriever (query) | {results['retriever']['query_encoding'].get('mean_ms', 'N/A'):.1f} |\n"
    if "reranker" in results and "rerank_24_candidates" in results["reranker"]:
        report += f"| Reranker (24 cand) | {results['reranker']['rerank_24_candidates'].get('mean_ms', 'N/A'):.1f} |\n"
    if "gnn" in results and "p3_graph_reranker" in results["gnn"]:
        report += f"| GNN (P3) | {results['gnn']['p3_graph_reranker'].get('mean_ms', 'N/A'):.1f} |\n"
    if "end_to_end" in results and "full_pipeline" in results["end_to_end"]:
        report += f"| **End-to-End** | **{results['end_to_end']['full_pipeline'].get('mean_ms', 'N/A'):.1f}** |\n"

    report += """
---

## Reproducibility

```bash
python scripts/analysis/measure_latency.py \\
    --output outputs/efficiency/ \\
    --n_samples 100 \\
    --warmup 5
```

## Notes

- All measurements include CUDA synchronization when GPU is used
- Warmup runs are excluded from timing
- Times are in milliseconds (ms)
- Throughput assumes single-query processing
"""

    with open(output_dir / "efficiency_report.md", "w") as f:
        f.write(report)

    logger.info(f"Report saved to {output_dir / 'efficiency_report.md'}")


def main():
    parser = argparse.ArgumentParser(description="Measure inference latency")
    parser.add_argument("--output", type=Path, default=Path("outputs/efficiency"))
    parser.add_argument("--n_samples", type=int, default=100, help="Number of timing runs")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup runs")
    parser.add_argument("--skip_retriever", action="store_true", help="Skip retriever measurement")
    parser.add_argument("--skip_reranker", action="store_true", help="Skip reranker measurement")
    parser.add_argument("--skip_gnn", action="store_true", help="Skip GNN measurement")
    parser.add_argument("--skip_e2e", action="store_true", help="Skip end-to-end measurement")

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("EFFICIENCY MEASUREMENT")
    logger.info(f"Samples: {args.n_samples}, Warmup: {args.warmup}")
    logger.info("=" * 80)

    # Get hardware info
    hardware = get_hardware_info()
    logger.info(f"Platform: {hardware.get('platform')}")
    logger.info(f"GPU: {hardware.get('gpu_name', 'N/A')}")

    results = {}

    # Measure each component
    if not args.skip_retriever:
        results["retriever"] = measure_retriever_latency(args.n_samples, args.warmup)

    if not args.skip_reranker:
        results["reranker"] = measure_reranker_latency(args.n_samples, args.warmup)

    if not args.skip_gnn:
        results["gnn"] = measure_gnn_latency(args.n_samples, args.warmup)

    if not args.skip_e2e:
        results["end_to_end"] = measure_end_to_end_latency(min(args.n_samples, 20), args.warmup)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "efficiency_results.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "hardware": hardware,
            "config": {
                "n_samples": args.n_samples,
                "warmup": args.warmup,
            },
            "results": results,
        }, f, indent=2, default=str)

    # Create CSV summary
    summary_rows = []
    for component, component_results in results.items():
        if isinstance(component_results, dict):
            for metric, data in component_results.items():
                if isinstance(data, dict) and "mean_ms" in data:
                    summary_rows.append({
                        "component": component,
                        "metric": metric,
                        "mean_ms": data["mean_ms"],
                        "std_ms": data.get("std_ms", 0),
                        "n_runs": data.get("n_runs", 0),
                    })

    if summary_rows:
        import pandas as pd
        pd.DataFrame(summary_rows).to_csv(output_dir / "efficiency_summary.csv", index=False)

    # Generate report
    generate_efficiency_report(results, hardware, output_dir)

    logger.info("=" * 80)
    logger.info("MEASUREMENT COMPLETE")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
