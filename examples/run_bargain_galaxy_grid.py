"""BARGAIN benchmark grid for the LLMSQL galaxy queries.

Mirrors a subset of LLMSQL's ``run_direct_galaxy_grid.py`` — the
queries the user wired into BARGAIN's evaluation suite — but routes
each query through :class:`BARGAIN_P` instead of the
sample-then-propagate pipeline. Per-query F1 / oracle-usage metrics
are written to a CSV for direct comparison against the LLMSQL grid
output.

Usage
-----
::

    # Run every wired-up query
    python run_bargain_galaxy_grid.py

    # Run a named group (matches the LLMSQL grid CLI conventions)
    python run_bargain_galaxy_grid.py ecomm
    python run_bargain_galaxy_grid.py match

    # Run a single query
    python run_bargain_galaxy_grid.py cars_q3

    # Sweep over multiple budgets / precision targets
    python run_bargain_galaxy_grid.py cars_q3 \\
        --budget-pcts 5 10 25 --target 0.9 --delta 0.1

Environment knobs
-----------------
``VLLM_BASE_URL``       OpenAI-compatible endpoint (default ``http://localhost:8000/v1``).
``BARGAIN_PROXY_MODEL`` Cheap model name (default ``Qwen/Qwen3-VL-2B-Instruct``).
``BARGAIN_ORACLE_MODEL`` Expensive model name (default ``Qwen/Qwen3-VL-30B-A3B-Instruct``).
``SEMBENCH_ROOT``       Path to ``SemBench/files`` (default ``/localhome/hza214/SemBench/files``).
``LROBENCH_DATABASES_DIR`` Path to LRobench databases (default ``./databases``).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd

# Allow running this file directly without pip-installing BARGAIN: the
# parent directory contains the top-level ``BARGAIN/`` package, and
# this file's own directory exposes the ``bargain_galaxy`` helper
# package alongside it (matching the convention used by the other
# example scripts in this folder, e.g. ``court_opinion_example.py``).
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))            # so ``import bargain_galaxy`` works
sys.path.insert(0, str(_HERE.parent))     # so ``from BARGAIN import ...`` works

from BARGAIN import BARGAIN_P  # noqa: E402
from bargain_galaxy.queries import (  # noqa: E402
    ALL_TAGS, GROUPS, Query, build_query_registry,
)
from bargain_galaxy.vllm_models import (  # noqa: E402
    VLLMOracle, VLLMProxy,
)


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


_FILTER_SYSTEM_PROMPT = (
    "You are a helpful data analyst. You will receive data and a query. "
    "Answer ONLY 'True' or 'False'."
)
_JOIN_SYSTEM_PROMPT = (
    "You are a helpful data analyst. You will receive data from two "
    "tables and a query. Answer ONLY 'True' or 'False'."
)


def _resolve_queries(filter_arg: str) -> List[str]:
    if filter_arg in ALL_TAGS:
        return [filter_arg]
    if filter_arg in GROUPS:
        return list(GROUPS[filter_arg])
    raise SystemExit(
        f"Unknown query/group {filter_arg!r}. "
        f"Choose from {sorted(ALL_TAGS)} or groups {sorted(GROUPS)}."
    )


def _budget_count(total: int, pct: float) -> int:
    """Convert a percentage into an integer oracle-call budget."""
    n = int(round(total * pct / 100.0))
    return max(1, n)


def _system_prompt_for(query: Query) -> str:
    return _JOIN_SYSTEM_PROMPT if query.kind == "join" else _FILTER_SYSTEM_PROMPT


def _run_single_query(
    query: Query,
    *,
    budget_pcts: Sequence[float],
    target: float,
    delta: float,
    proxy_model: str,
    oracle_model: str,
    base_url: str,
    seed: int,
    proxy_workers: int,
    oracle_workers: int,
) -> List[dict]:
    print(f"\n{'='*70}\n[{query.tag}] kind={query.kind}  modality={query.modality}\n{'='*70}")
    t0 = time.time()
    records, keys = query.build_records()
    n = len(records)
    print(f"[{query.tag}] built {n} data records in {time.time()-t0:.1f}s")

    if n == 0:
        print(f"[{query.tag}] no records — skipping")
        return []

    rows: List[dict] = []
    system_prompt = _system_prompt_for(query)

    for pct in budget_pcts:
        budget = min(_budget_count(n, pct), n)
        print(
            f"\n[{query.tag}] >>> budget_pct={pct} -> {budget} oracle calls "
            f"(target={target} delta={delta})"
        )

        proxy = VLLMProxy(
            model=proxy_model, base_url=base_url,
            system_prompt=system_prompt, max_workers=proxy_workers,
        )
        oracle = VLLMOracle(
            model=oracle_model, base_url=base_url,
            system_prompt=system_prompt, max_workers=oracle_workers,
        )

        bargain = BARGAIN_P(
            proxy=proxy, oracle=oracle,
            delta=delta, target=target, budget=budget, seed=seed,
        )
        start = time.time()
        positive_idxs = bargain.process(records)
        elapsed = time.time() - start

        predicted_keys = [keys[i] for i in positive_idxs]
        print(
            f"[{query.tag}] BARGAIN_P returned {len(positive_idxs)} positives "
            f"in {elapsed:.1f}s; scoring..."
        )
        try:
            precision, recall, f1 = query.score(predicted_keys)
        except Exception as exc:  # noqa: BLE001 — keep the loop alive
            logger.exception("[%s] scorer failed: %s", query.tag, exc)
            precision = recall = f1 = float("nan")

        oracle_calls = oracle.get_number_preds()
        rows.append({
            "query": query.tag,
            "kind": query.kind,
            "modality": query.modality,
            "n_records": n,
            "budget_pct": pct,
            "budget": budget,
            "target": target,
            "delta": delta,
            "n_positives": len(positive_idxs),
            "oracle_calls": int(oracle_calls),
            "oracle_frac": float(oracle_calls) / max(n, 1),
            "elapsed_sec": round(elapsed, 2),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    return rows


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run BARGAIN_P over the LLMSQL galaxy-grid queries.",
    )
    parser.add_argument(
        "filter", nargs="?", default="all",
        help="Query tag or group (default 'all').",
    )
    parser.add_argument(
        "--budget-pcts", nargs="+", type=float,
        default=[5.0, 10.0, 25.0],
        help="Oracle-call budget as percentages of dataset size.",
    )
    parser.add_argument("--target", type=float, default=0.9,
                        help="Precision target for BARGAIN_P.")
    parser.add_argument("--delta", type=float, default=0.1,
                        help="Allowed failure probability for BARGAIN_P.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (forwarded to BARGAIN_P / numpy).")
    parser.add_argument(
        "--proxy-model",
        default=os.environ.get("BARGAIN_PROXY_MODEL", "Qwen/Qwen3-VL-2B-Instruct"),
    )
    parser.add_argument(
        "--oracle-model",
        default=os.environ.get("BARGAIN_ORACLE_MODEL", "Qwen/Qwen3-VL-30B-A3B-Instruct"),
    )
    parser.add_argument(
        "--vllm-url",
        default=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
        help="OpenAI-compatible vLLM endpoint base URL.",
    )
    parser.add_argument("--proxy-workers", type=int, default=32,
                        help="Concurrent in-flight proxy requests.")
    parser.add_argument("--oracle-workers", type=int, default=16,
                        help="Concurrent in-flight oracle requests.")
    parser.add_argument(
        "--results-dir",
        default=os.environ.get("BARGAIN_RESULTS_DIR", "./results/bargain_grid"),
    )
    args = parser.parse_args(argv)

    os.makedirs(args.results_dir, exist_ok=True)
    np.random.seed(args.seed)

    registry = build_query_registry(args.results_dir)
    tags = _resolve_queries(args.filter)

    print(f"Running BARGAIN grid for: {tags}")
    print(
        f"  vLLM URL : {args.vllm_url}\n"
        f"  proxy    : {args.proxy_model}\n"
        f"  oracle   : {args.oracle_model}\n"
        f"  budgets  : {args.budget_pcts} (% of dataset)\n"
        f"  target/delta : {args.target}/{args.delta}"
    )

    all_rows: List[dict] = []
    for tag in tags:
        if tag not in registry:
            print(f"[{tag}] not registered — skipping")
            continue
        try:
            rows = _run_single_query(
                registry[tag],
                budget_pcts=args.budget_pcts,
                target=args.target, delta=args.delta,
                proxy_model=args.proxy_model,
                oracle_model=args.oracle_model,
                base_url=args.vllm_url,
                seed=args.seed,
                proxy_workers=args.proxy_workers,
                oracle_workers=args.oracle_workers,
            )
        except Exception as exc:  # noqa: BLE001 — keep going to next query
            logger.exception("[%s] failed: %s", tag, exc)
            rows = []
        all_rows.extend(rows)
        # Persist after every query so a crash mid-grid still leaves
        # partial results on disk.
        if rows:
            per_query_path = os.path.join(args.results_dir, f"{tag}.csv")
            pd.DataFrame(rows).to_csv(per_query_path, index=False)
            print(f"[{tag}] wrote {per_query_path}")

    if not all_rows:
        print("No results produced.")
        return 1

    summary_path = os.path.join(args.results_dir, "summary.csv")
    summary_df = pd.DataFrame(all_rows)
    summary_df.to_csv(summary_path, index=False)
    print(f"\nWrote summary: {summary_path}")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.4f}".format)
    print("\nBARGAIN GRID SUMMARY")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
