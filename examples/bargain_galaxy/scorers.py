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


def score_cars_q4(predicted_ids: Iterable[str], gt_path: str) -> Tuple[float, float, float]:
    """Row-level F1 for Q4 (engine-complaint filter).

    Scores the predicted positive car_ids against the per-row ground truth
    at ``gt_path`` (``Q4_ground_truth_rows_sf9836.csv`` — one positive row
    per engine-complaint car): precision / recall / F1 over car_id sets,
    mirroring the other set-based filter scorers. Replaces the legacy
    average-age relative-error proxy so F1 is the run's quality metric.

    Diagnostic note: the printed GT / Pred / TP counts make a dataset
    scale-factor mismatch obvious — if the predicted car_ids come from a
    different scale factor than the ground-truth rows, TP collapses to ~0.
    """
    try:
        gt_df = pd.read_csv(gt_path, dtype=str)
    except Exception as e:  # noqa: BLE001 — degrade to F1=0 on read failure
        print(f"  [cars_q4] Error reading ground truth: {e}")
        return 0.0, 0.0, 0.0
    gt_col = "car_id" if "car_id" in gt_df.columns else gt_df.columns[0]
    gt_ids = {
        cid for cid in (_normalize_car_id(v) for v in gt_df[gt_col].dropna())
        if cid is not None
    }

    pred_ids = {
        cid for cid in (_normalize_car_id(v) for v in predicted_ids)
        if cid is not None
    }

    tp = len(gt_ids & pred_ids)
    fp = len(pred_ids - gt_ids)
    fn = len(gt_ids - pred_ids)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    print(
        f"  [cars_q4] GT: {len(gt_ids)}, Pred: {len(pred_ids)}, "
        f"TP={tp} FP={fp} FN={fn}"
    )
    print(f"  [cars_q4] P={precision:.4f} R={recall:.4f} F1={f1:.4f}")
    return precision, recall, f1


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


# --- movie q5 / q7: unordered review-pair F1 ---------------------------
def score_movie_pairs(
    predicted_keys: Iterable[Sequence], gt_path: str,
) -> Tuple[float, float, float]:
    """Set-based F1 over unordered ``(movie_id, {reviewId_a, reviewId_b})``
    pairs.

    ``predicted_keys`` are ``(left_reviewId, movie_id, right_reviewId)``
    triples (the key layout produced by the movie self-join builder).
    Reflexive pairs (a review compared with itself) are dropped, and the
    two review ids are sorted so ``(a, b)`` and ``(b, a)`` collapse to a
    single pair. The ground-truth CSV stores one pair per row with
    columns ``[movie_id, reviewId_a, reviewId_b]`` (matching the LLMSQL
    grid's ``calculate_pair_f1`` layout).
    """
    sys_pairs: Set = set()
    for key in predicted_keys:
        rid1, movie_id, rid2 = str(key[0]).strip(), str(key[1]).strip(), str(key[2]).strip()
        if rid1 == rid2:
            continue
        sys_pairs.add((movie_id, tuple(sorted([rid1, rid2]))))

    gt_df = pd.read_csv(gt_path)
    cols = list(gt_df.columns)
    gt_pairs: Set = set()
    for _, row in gt_df.iterrows():
        if len(cols) < 3:
            continue
        mid, v1, v2 = row[cols[0]], row[cols[1]], row[cols[2]]
        if pd.notna(mid) and pd.notna(v1) and pd.notna(v2):
            gt_pairs.add((str(mid).strip(), tuple(sorted([str(v1).strip(), str(v2).strip()]))))

    p, r, f1 = _set_f1(sys_pairs, gt_pairs)
    print(f"  [movie] GT={len(gt_pairs)} Pred={len(sys_pairs)} P={p:.4f} R={r:.4f} F1={f1:.4f}")
    return p, r, f1


# --- classify (sem_join images/rows × categories) macro F1 -------------
def _norm_category(value) -> str | None:
    if value is None:
        return None
    s = str(value).strip().lower().replace("\n", "")
    if s in ("", "nan"):
        return None
    return s


def score_classify_macro_f1(
    predicted_keys: Iterable[Sequence],
    gt_path: str,
    *,
    id_col: str,
    cat_col: str,
    fallback_category: str,
) -> Tuple[float, float, float]:
    """Macro-averaged precision/recall/F1 for a CLASSIFY-style join.

    A classify query is decomposed into ``(record × category)`` pairs;
    BARGAIN returns the positive pairs. For each record we keep the first
    positive category as its predicted class; records with no positive
    pair fall back to ``fallback_category``. We then compare against the
    per-record ground-truth class and macro-average the per-class F1 over
    the union of classes present in the truth or predictions (matching
    sklearn's ``average='macro'`` with ``zero_division=0``).

    ``predicted_keys`` are ``(record_id, category)`` tuples.
    """
    pred_dict: dict = {}
    for key in predicted_keys:
        rid = str(key[0]).strip()
        cat = _norm_category(key[1])
        if cat is None:
            continue
        pred_dict.setdefault(rid, cat)  # keep the first positive category

    gt_df = pd.read_csv(gt_path, dtype=str)
    gt_df.columns = [c.strip().lower() for c in gt_df.columns]
    id_c = id_col.strip().lower()
    cat_c = cat_col.strip().lower()
    if id_c not in gt_df.columns or cat_c not in gt_df.columns:
        id_c, cat_c = gt_df.columns[0], gt_df.columns[1]

    fb = _norm_category(fallback_category)
    y_true: list = []
    y_pred: list = []
    n_fallback = 0
    for _, row in gt_df.iterrows():
        rid = str(row[id_c]).strip()
        gt_cat = _norm_category(row[cat_c])
        if gt_cat is None:
            continue
        y_true.append(gt_cat)
        if rid in pred_dict:
            y_pred.append(pred_dict[rid])
        else:
            y_pred.append(fb)
            n_fallback += 1

    labels = sorted(set(y_true) | set(y_pred))
    ps: list = []
    rs: list = []
    fs: list = []
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        ps.append(p)
        rs.append(r)
        fs.append(f)
    macro_p = sum(ps) / len(ps) if ps else 0.0
    macro_r = sum(rs) / len(rs) if rs else 0.0
    macro_f1 = sum(fs) / len(fs) if fs else 0.0
    print(
        f"  [classify] GT={len(y_true)} matched={len(y_true) - n_fallback} "
        f"fallback={n_fallback} (-> '{fallback_category}') "
        f"P={macro_p:.4f} R={macro_r:.4f} F1={macro_f1:.4f}"
    )
    return macro_p, macro_r, macro_f1
