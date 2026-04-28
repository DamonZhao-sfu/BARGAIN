"""F1 / aggregation scorers ported from the LLMSQL galaxy grid.

Each scorer takes (predicted_ids, ground_truth_path) and returns a
``(precision, recall, f1)`` triple. Multimodal queries that compare
against an in-memory hardcoded GT (mmqa_q2a, mmqa_q7) ignore the path
argument. Aggregation queries (animals_q1, cars_q4) return the legacy
``1 - min(rel_err, 1)`` proxy of F1 so the same column makes sense
across all scorers.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence, Set, Tuple

import pandas as pd


def _set_f1(predicted: Set, gt: Set) -> Tuple[float, float, float]:
    tp = len(predicted & gt)
    fp = len(predicted - gt)
    fn = len(gt - predicted)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def score_id_set(predicted_ids: Iterable[str], gt_path: str) -> Tuple[float, float, float]:
    """Generic set-based F1 over a single id column."""
    gt_df = pd.read_csv(gt_path)
    gt = set(str(x).strip() for x in gt_df.iloc[:, 0].tolist())
    pred = set(str(x).strip() for x in predicted_ids)
    p, r, f1 = _set_f1(pred, gt)
    print(f"  GT={len(gt)} Pred={len(pred)} P={p:.4f} R={r:.4f} F1={f1:.4f}")
    return p, r, f1


def score_id_set_topk(
    predicted_ids: Iterable[str], gt_path: str, k: int = 100,
) -> Tuple[float, float, float]:
    """Top-K set F1 (sort lexicographically, truncate, then F1).

    Used for ecomm_q7 where the full self-join GT is huge and we only
    care about the highest-ranked slice.
    """
    gt_df = pd.read_csv(gt_path)
    pred_sorted = sorted(str(x).strip() for x in predicted_ids)
    gt_sorted = sorted(str(x).strip() for x in gt_df.iloc[:, 0].tolist())
    if k and k > 0:
        pred_top = set(pred_sorted[:k])
        gt_top = set(gt_sorted[:k])
    else:
        pred_top = set(pred_sorted)
        gt_top = set(gt_sorted)
    p, r, f1 = _set_f1(pred_top, gt_top)
    print(
        f"  [top{k}] GT={len(gt_top)}/{len(gt_sorted)} "
        f"Pred={len(pred_top)}/{len(pred_sorted)} "
        f"P={p:.4f} R={r:.4f} F1={f1:.4f}"
    )
    return p, r, f1


# --- mmqa hardcoded ground truths (verbatim from the LLMSQL grid) ---
MMQA_Q2A_GT = {
    (0, "117d500aaa630023c4038b8268b309c0.png"),
    (5, "117d500aaa630023c4038b8268b309c0.png"),
    (6, "117d500aaa630023c4038b8268b309c0.png"),
    (9, "117d500aaa630023c4038b8268b309c0.png"),
    (10, "117d500aaa630023c4038b8268b309c0.png"),
}

MMQA_Q7_GT = {
    ("British Airways", "cwncgabxti09zmf36t4phvaz1xzv10wu.png"),
    ("Delta Air Lines", "1sz0kf8wcmj0q8n3pu6mg61gl158vvz1.png"),
    ("Discover Airlines", "nqr1pjql5qs3dp8rz2a4zzq7rn0xyp6k.png"),
    ("Edelweiss Air", "6zpijbg5jv4jftpftpmla6xbizne28d9.png"),
    ("Virgin Atlantic", "antbx4oxst0z5o6pe2g1thrjr0dz073j.png"),
}


def score_mmqa_q2a(predicted_pairs: Iterable[Sequence], _gt_path_unused: str = "") -> Tuple[float, float, float]:
    pred = set()
    for pair in predicted_pairs:
        try:
            pred.add((int(pair[0]), str(pair[1]).strip()))
        except (ValueError, TypeError):
            continue
    p, r, f1 = _set_f1(pred, MMQA_Q2A_GT)
    print(f"  [mmqa_q2a] GT={len(MMQA_Q2A_GT)} Pred={len(pred)} P={p:.4f} R={r:.4f} F1={f1:.4f}")
    return p, r, f1


def score_mmqa_q7(predicted_pairs: Iterable[Sequence], _gt_path_unused: str = "") -> Tuple[float, float, float]:
    pred = set()
    for pair in predicted_pairs:
        pred.add((str(pair[0]).strip(), str(pair[1]).strip()))
    p, r, f1 = _set_f1(pred, MMQA_Q7_GT)
    print(f"  [mmqa_q7] GT={len(MMQA_Q7_GT)} Pred={len(pred)} P={p:.4f} R={r:.4f} F1={f1:.4f}")
    return p, r, f1


# --- cars q3 / q8: SemBench limit-aware GT sampling --------------------
def _normalize_car_id(val) -> str | None:
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return None
    if re.fullmatch(r"\d+\.0", s):
        s = s.split(".")[0]
    if s.isdigit():
        return s
    return None


def _limit_aware_score(
    predicted_ids: Iterable[str],
    gt_path: str,
    id_column: str,
    limit: int,
    normalize=lambda v: str(v).strip(),
) -> Tuple[float, float, float]:
    gt_df = pd.read_csv(gt_path, dtype=str)
    if id_column not in gt_df.columns:
        gt_df = gt_df.rename(columns={gt_df.columns[0]: id_column})
    gt_df[id_column] = gt_df[id_column].apply(normalize)
    gt_df = gt_df.dropna(subset=[id_column])

    predicted = [normalize(v) for v in predicted_ids]
    predicted = [p for p in predicted if p is not None]
    if len(predicted) > limit:
        predicted = predicted[:limit]
    pred_set = set(predicted)

    correct_mask = gt_df[id_column].isin(list(pred_set))
    correct = gt_df.loc[correct_mask]
    if correct.empty:
        n = min(limit, len(gt_df))
        sample = gt_df.sample(n=n, random_state=42) if n else gt_df
    elif len(correct) >= limit:
        sample = correct.head(limit)
    else:
        false_cases = gt_df[~correct_mask]
        n_needed = min(limit - len(correct), len(false_cases))
        if n_needed > 0:
            sample = pd.concat(
                [correct, false_cases.sample(n=n_needed, random_state=42)]
            )
        else:
            sample = correct

    gt_set = set(sample[id_column].dropna().astype(str).str.strip())
    p, r, f1 = _set_f1(pred_set, gt_set)
    print(
        f"  GT_sample={len(gt_set)} (LIMIT={limit}) Pred={len(pred_set)} "
        f"P={p:.4f} R={r:.4f} F1={f1:.4f}"
    )
    return p, r, f1


def score_cars_q8(predicted_ids: Iterable[str], gt_path: str) -> Tuple[float, float, float]:
    return _limit_aware_score(
        predicted_ids, gt_path, id_column="car_id", limit=100,
        normalize=_normalize_car_id,
    )


def score_cars_q3(predicted_vins: Iterable[str], gt_path: str) -> Tuple[float, float, float]:
    return _limit_aware_score(
        predicted_vins, gt_path, id_column="vin", limit=341,
        normalize=lambda v: str(v).strip() or None,
    )


def score_cars_q4(
    predicted_ids: Iterable[str], gt_path: str,
    cars_csv: str | None = None,
) -> Tuple[float, float, float]:
    """Q4 evaluates a scalar (avg car age over engine-problem cars).

    Returns ``1 - min(relative_error, 1)`` as the F1 proxy so the
    column type matches the other scorers.
    """
    pred_ids = {int(p) for p in (_normalize_car_id(v) for v in predicted_ids) if p}
    if not pred_ids:
        print(f"  [cars_q4] no predicted ids — F1=0")
        return 0.0, 0.0, 0.0

    if cars_csv is None:
        # Fall back to the canonical SemBench location used by LLMSQL.
        cars_csv = "/localhome/hza214/SemBench/files/cars/data/sf_157376/car_data_157376.csv"

    cars_df = pd.read_csv(cars_csv)
    matched = cars_df[cars_df["car_id"].isin(pred_ids)]
    if matched.empty:
        print(f"  [cars_q4] no predicted ids matched cars table — F1=0")
        return 0.0, 0.0, 0.0
    pred_avg_age = float(2026 - matched["year"].mean())

    scalar_gt = pd.read_csv(gt_path)
    gt_avg_age = float(scalar_gt.iloc[0, 0])
    abs_err = abs(pred_avg_age - gt_avg_age)
    rel_err = abs_err / max(abs(gt_avg_age), 1e-12)
    score = 1.0 - min(rel_err, 1.0)
    print(
        f"  [cars_q4] predicted_count={len(pred_ids)} "
        f"pred_avg_age={pred_avg_age:.4f} gt_avg_age={gt_avg_age:.4f} "
        f"rel_err={rel_err:.4f} score={score:.4f}"
    )
    return score, score, score


def score_animals_q1(predicted_ids: Iterable[str], gt_path: str) -> Tuple[float, float, float]:
    """Animals Q1 is a scalar count comparison (zebra count vs GT)."""
    predicted_count = sum(1 for _ in predicted_ids)
    gt_df = pd.read_csv(gt_path)
    gt_count = int(gt_df.iloc[0, 0])
    abs_err = abs(predicted_count - gt_count)
    rel_err = abs_err / max(abs(gt_count), 1e-12)
    score = 1.0 - min(rel_err, 1.0)
    print(
        f"  [animals_q1] gt_count={gt_count} predicted={predicted_count} "
        f"rel_err={rel_err:.4f} score={score:.4f}"
    )
    return score, score, score
