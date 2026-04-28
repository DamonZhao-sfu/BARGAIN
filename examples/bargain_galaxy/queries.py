"""Per-query data loaders, prompts, and scorers for the BARGAIN grid.

Each query is registered under a string tag (matching the LLMSQL grid
naming so log lines and result files line up). A query exposes:

* :attr:`Query.kind`      — ``"filter"`` or ``"join"`` (controls which
  system prompt the proxy/oracle uses).
* :attr:`Query.modality`  — ``"text"`` or ``"multimodal"``; multimodal
  queries route image columns into the chat request as ``image_url``
  blocks.
* :meth:`Query.build_records` — return the list of
  :class:`BargainRecord` objects that BARGAIN will sample over.
* :meth:`Query.score` — compute (precision, recall, F1) given the
  predicted positive ids and the configured ground-truth path.

Data is loaded with pandas (no Spark dependency); the source CSVs /
parquets follow the same layout LLMSQL uses, so the same SemBench /
LRobench data dirs can be re-used unchanged.
"""

from __future__ import annotations

import itertools
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from bargain_galaxy import scorers
from bargain_galaxy.vllm_models import BargainRecord


SEMBENCH_ROOT = os.environ.get(
    "SEMBENCH_ROOT", "/localhome/hza214/SemBench/files",
)
LROBENCH_DATABASES_DIR = os.environ.get(
    "LROBENCH_DATABASES_DIR", "./databases",
)


# ---------------------------------------------------------------------------
# Lightweight prompt rendering (mirrors proxy_model._replace_placeholder)
# ---------------------------------------------------------------------------
import re

_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")
_TYPED_PLACEHOLDER_RE = re.compile(r"\{(\w+):(\w+)\}")


def _typed_columns(prompt: str) -> Dict[str, str]:
    """Return ``{col_name: col_type}`` for ``{type:col}`` placeholders."""
    return {col: ctype for ctype, col in _TYPED_PLACEHOLDER_RE.findall(prompt)}


def _all_columns(prompt: str) -> List[str]:
    return _PLACEHOLDER_RE.findall(prompt) + list(_typed_columns(prompt).keys())


def _render(prompt: str, values: Dict[str, Any], image_columns: Sequence[str]) -> Tuple[str, Tuple[str, ...]]:
    """Replace placeholders in ``prompt`` with ``values``.

    Image columns are stripped from the rendered text (their value is
    returned in the ``images`` tuple instead, so the proxy/oracle can
    attach them as separate ``image_url`` content blocks).
    """
    text = prompt
    images: List[str] = []
    for col in _all_columns(prompt):
        val = values.get(col, "")
        if col in image_columns:
            sval = str(val).strip()
            if sval:
                images.append(sval)
            replacement = ""
        else:
            replacement = "" if val is None else str(val)
        text = re.sub(r"\{\w+:" + re.escape(col) + r"\}", lambda _: replacement, text)
        text = text.replace("{" + col + "}", replacement)
    return text, tuple(images)


# ---------------------------------------------------------------------------
# Query base classes
# ---------------------------------------------------------------------------
@dataclass
class Query:
    tag: str
    kind: str  # "filter" or "join"
    modality: str  # "text" or "multimodal"
    prompt: str
    gt_path: Optional[str]
    score_fn: Callable
    build_fn: Callable[[], Tuple[List[BargainRecord], List[Tuple[Any, ...]]]]
    # ``id_components`` describes how the BargainRecord.id maps back
    # into a (multi-)column tuple for scoring; e.g. ("car_id",) for
    # cars_q3 or ("ID", "image_filename") for mmqa_q2a.
    id_components: Tuple[str, ...] = ("id",)

    def build_records(self) -> Tuple[List[BargainRecord], List[Tuple[Any, ...]]]:
        return self.build_fn()

    def score(self, predicted_keys: Iterable[Tuple[Any, ...]]) -> Tuple[float, float, float]:
        return self.score_fn(predicted_keys, self.gt_path)


# ---------------------------------------------------------------------------
# Helpers for building per-row / per-pair records
# ---------------------------------------------------------------------------
def _filter_records(
    df: pd.DataFrame, prompt: str, id_columns: Sequence[str],
) -> Tuple[List[BargainRecord], List[Tuple[Any, ...]]]:
    img_cols = [c for c, t in _typed_columns(prompt).items() if t == "image"]
    records: List[BargainRecord] = []
    keys: List[Tuple[Any, ...]] = []
    for _, row in df.iterrows():
        values = row.to_dict()
        text, images = _render(prompt, values, img_cols)
        key = tuple(row[c] for c in id_columns)
        rid = "|".join(str(v) for v in key)
        records.append(BargainRecord(id=rid, prompt=text, images=images))
        keys.append(key)
    return records, keys


def _join_records(
    left: pd.DataFrame, right: pd.DataFrame,
    prompt: str, left_id_cols: Sequence[str], right_id_cols: Sequence[str],
    *, drop_self_join: bool = False,
) -> Tuple[List[BargainRecord], List[Tuple[Any, ...]]]:
    """Cross-join two pandas dataframes into BargainRecord pairs."""
    img_cols = [c for c, t in _typed_columns(prompt).items() if t == "image"]
    records: List[BargainRecord] = []
    keys: List[Tuple[Any, ...]] = []

    left_rows = left.to_dict("records")
    right_rows = right.to_dict("records")

    for lrow, rrow in itertools.product(left_rows, right_rows):
        if drop_self_join:
            # Equality on the first id column is the canonical definition
            # in the LLMSQL grid (ecomm_q9 self-join).
            if lrow[left_id_cols[0]] == rrow[right_id_cols[0]]:
                continue
        merged = {**lrow, **rrow}
        text, images = _render(prompt, merged, img_cols)
        key = tuple(lrow[c] for c in left_id_cols) + tuple(rrow[c] for c in right_id_cols)
        rid = "-".join(str(v) for v in key)
        records.append(BargainRecord(id=rid, prompt=text, images=images))
        keys.append(key)
    return records, keys


# ---------------------------------------------------------------------------
# Query builders (data loaders + prompt definitions)
# ---------------------------------------------------------------------------
def _ecomm_styles(data_dir: str) -> pd.DataFrame:
    styles = pd.read_parquet(os.path.join(data_dir, "styles_details.parquet"))
    if "productDescriptors" in styles.columns:
        # The nested struct layout matches the parquet schema LLMSQL uses.
        try:
            styles["description_text"] = styles["productDescriptors"].apply(
                lambda d: d.get("description", {}).get("value", "")
                if isinstance(d, dict) else "",
            )
        except AttributeError:
            styles["description_text"] = ""
    drop_cols = [c for c in styles.columns if c not in (
        "id", "productDisplayName", "description_text", "price",
        "baseColour", "colour1", "colour2",
    )]
    return styles.drop(columns=[c for c in drop_cols if c in styles.columns])


def _build_ecomm_q7() -> Tuple[List[BargainRecord], List[Tuple[Any, ...]]]:
    data_dir = os.path.join(SEMBENCH_ROOT, "ecomm/data/sf_4000")
    styles = _ecomm_styles(data_dir)
    if "price" in styles.columns:
        styles = styles[styles["price"] <= 500]
    left = styles.rename(columns={
        "id": "left_id",
        "productDisplayName": "left_productDisplayName",
        "description_text": "left_description_text",
    })[["left_id", "left_productDisplayName", "left_description_text"]]
    right = styles.rename(columns={
        "id": "right_id",
        "productDisplayName": "right_productDisplayName",
        "description_text": "right_description_text",
    })[["right_id", "right_productDisplayName", "right_description_text"]]

    prompt = (
        "You will be given two product descriptions. Do both product descriptions "
        "describe products of the same category from the same brand, e.g., both "
        "are t-shirts from Adidas?\n\n"
        "The first product description is:\n"
        "{left_productDisplayName} - {left_description_text}\n\n"
        "The second product description is:\n"
        "{right_productDisplayName} - {right_description_text}"
    )
    return _join_records(
        left, right, prompt,
        left_id_cols=("left_id",), right_id_cols=("right_id",),
    )


def _build_ecomm_q8() -> Tuple[List[BargainRecord], List[Tuple[Any, ...]]]:
    data_dir = os.path.join(SEMBENCH_ROOT, "ecomm/data/sf_4000")
    styles = _ecomm_styles(data_dir)
    image_map = pd.read_parquet(os.path.join(data_dir, "image_mapping.parquet"))
    styles = styles[styles["description_text"].fillna("").str.len() >= 3000]

    left = styles.rename(columns={
        "id": "style_id",
        "productDisplayName": "left_productDisplayName",
        "description_text": "left_description_text",
    })[["style_id", "left_productDisplayName", "left_description_text"]]
    image_map["right_image_filepath"] = image_map["filename"].apply(
        lambda f: os.path.join(data_dir, "images", str(f))
    )
    right = image_map.rename(columns={"id": "image_id"})[["image_id", "right_image_filepath"]]

    prompt = (
        "Does the image fit the product description?\n"
        "Image: {image:right_image_filepath}\n"
        "Product name: {left_productDisplayName}\n"
        "Product description: {left_description_text}"
    )
    return _join_records(
        left, right, prompt,
        left_id_cols=("style_id",), right_id_cols=("image_id",),
    )


def _build_ecomm_q9() -> Tuple[List[BargainRecord], List[Tuple[Any, ...]]]:
    data_dir = os.path.join(SEMBENCH_ROOT, "ecomm/data/sf_4000")
    styles = _ecomm_styles(data_dir)
    image_map = pd.read_parquet(os.path.join(data_dir, "image_mapping.parquet"))

    valid_colors = {"Black", "Blue", "Red", "White", "Orange", "Green"}
    styles = styles[
        styles["baseColour"].isin(valid_colors)
        & (styles["colour1"].fillna("") == "")
        & (styles["colour2"].fillna("") == "")
        & (styles["price"] < 800)
    ]
    keep_ids = set(styles["id"].astype(str))
    image_map = image_map[image_map["id"].astype(str).isin(keep_ids)]
    image_map["image_filepath"] = image_map["filename"].apply(
        lambda f: os.path.join(data_dir, "images", str(f))
    )
    left = image_map.rename(columns={
        "id": "left_id", "image_filepath": "left_image_filepath",
    })[["left_id", "left_image_filepath"]]
    right = image_map.rename(columns={
        "id": "right_id", "image_filepath": "right_image_filepath",
    })[["right_id", "right_image_filepath"]]

    prompt = (
        "Determine whether both images display objects of the same category "
        "(e.g., both are shoes, both are bags, etc.) and whether these objects "
        "share the same dominant surface color. Disregard any logos, text, or "
        "printed graphics on the objects. There might be other objects in the "
        "images. Only focus on the main object. Base your comparison solely on "
        "object type and overall surface color.\n"
        "Left image: {image:left_image_filepath}\n"
        "Right image: {image:right_image_filepath}"
    )
    return _join_records(
        left, right, prompt,
        left_id_cols=("left_id",), right_id_cols=("right_id",),
        drop_self_join=True,
    )


def _build_ecomm_q13() -> Tuple[List[BargainRecord], List[Tuple[Any, ...]]]:
    data_dir = os.path.join(SEMBENCH_ROOT, "ecomm/data/sf_4000")
    styles = _ecomm_styles(data_dir)
    image_map = pd.read_parquet(os.path.join(data_dir, "image_mapping.parquet"))
    styles["id_str"] = styles["id"].astype(str)
    image_map["image_id_str"] = image_map["id"].astype(str)
    merged = styles.merge(
        image_map[["image_id_str", "filename"]],
        left_on="id_str", right_on="image_id_str", how="inner",
    )
    merged["image_filepath"] = merged["filename"].apply(
        lambda f: os.path.join(data_dir, "images", str(f))
    )
    df = merged[["id", "productDisplayName", "description_text", "image_filepath"]]

    prompt = (
        "You will receive a description of what a customer is looking for "
        "together with an image and a textual description of the product.\n"
        "Determine if they both match.\n\n"
        "I am looking for a running shirt for men with a round neck and short "
        "sleeves, preferably in blue or black, but not bright colors like "
        "white. Also definitely not green. It should be suitable for outdoor "
        "running in warm weather. If the t-shirt is not green, it should at "
        "least feature a striped design.\n\n"
        "The product has the following image {image:image_filepath} and "
        "textual description {productDisplayName} {description_text}"
    )
    return _filter_records(df, prompt, id_columns=("id",))


def _build_mmqa_q2a() -> Tuple[List[BargainRecord], List[Tuple[Any, ...]]]:
    data_dir = os.path.join(SEMBENCH_ROOT, "mmqa/data/sf_800")
    ap = pd.read_csv(os.path.join(data_dir, "ap_warrior.csv"))
    images = pd.read_csv(os.path.join(data_dir, "thalamusdb_images.csv"))
    images["image_filepath"] = images["image_filename"].apply(
        lambda f: os.path.join(data_dir, "images", str(f))
    )
    prompt = (
        "You will be provided with a horse racetrack name and an image.\n"
        "Determine if the image shows the logo of the racetrack and A.P. "
        "Warrior was a contender.\n"
        "Racetrack: {text:Track}\n"
        "Image: {image:image_filepath}"
    )
    return _join_records(
        ap, images, prompt,
        left_id_cols=("ID",), right_id_cols=("image_filename",),
    )


def _build_mmqa_q7() -> Tuple[List[BargainRecord], List[Tuple[Any, ...]]]:
    data_dir = os.path.join(SEMBENCH_ROOT, "mmqa/data/sf_800")
    airport = pd.read_csv(os.path.join(data_dir, "tampa_international_airport.csv"))
    airport = airport.drop_duplicates(subset=["Airlines"])
    images = pd.read_csv(os.path.join(data_dir, "thalamusdb_images.csv"))
    images["image_filepath"] = images["image_filename"].apply(
        lambda f: os.path.join(data_dir, "images", str(f))
    )
    prompt = (
        "You will be provided with an airline name {text:Airlines} and its "
        "destination {text:Destinations} and an image {image:image_filepath}.\n"
        "Determine for each airline if the airline's destination is in "
        "Europe and if the image shows the logo of the airline."
    )
    return _join_records(
        airport, images, prompt,
        left_id_cols=("Airlines",), right_id_cols=("image_filename",),
    )


def _build_cars_q3() -> Tuple[List[BargainRecord], List[Tuple[Any, ...]]]:
    data_dir = os.path.join(SEMBENCH_ROOT, "cars/data/sf_157376")
    car = pd.read_csv(os.path.join(data_dir, "car_data_157376.csv"))
    image = pd.read_csv(os.path.join(data_dir, "image_car_data_157376.csv"))
    image["image_path"] = image["image_path"].apply(
        lambda p: os.path.join(SEMBENCH_ROOT.replace("/files", ""), str(p))
        if not str(p).startswith("/") else str(p)
    )
    car = car[car["transmission"] == "Manual"]
    merged = car.merge(image, on="car_id", how="inner")
    df = merged[["vin", "image_path"]]

    prompt = (
        "You are given an image of a vehicle or its parts. Return true if the "
        "car is NOT damaged (pristine / no visible damage). "
        "Image: {image:image_path}"
    )
    return _filter_records(df, prompt, id_columns=("vin",))


def _build_cars_q4() -> Tuple[List[BargainRecord], List[Tuple[Any, ...]]]:
    data_dir = os.path.join(SEMBENCH_ROOT, "cars/data/sf_157376")
    complaints = pd.read_csv(os.path.join(data_dir, "text_complaints_data_157376.csv"))
    df = complaints[["car_id", "summary"]]
    prompt = (
        "You are given a textual complaint about a vehicle. Return true if "
        "the complaint describes a problem with the car's engine (or a "
        "component directly connected to the engine), false otherwise.\n"
        "Complaint: {summary}"
    )
    return _filter_records(df, prompt, id_columns=("car_id",))


def _build_cars_q8() -> Tuple[List[BargainRecord], List[Tuple[Any, ...]]]:
    data_dir = os.path.join(SEMBENCH_ROOT, "cars/data/sf_157376")
    car = pd.read_csv(os.path.join(data_dir, "car_data_157376.csv"))
    image = pd.read_csv(os.path.join(data_dir, "image_car_data_157376.csv"))
    image["image_path"] = image["image_path"].apply(
        lambda p: os.path.join(SEMBENCH_ROOT.replace("/files", ""), str(p))
        if not str(p).startswith("/") else str(p)
    )
    merged = car.merge(image, on="car_id", how="inner")
    df = merged[["car_id", "image_path"]]
    prompt = (
        "You are given an image of a vehicle or its parts. Return true if "
        "car has both, puncture and paint scratches. Image: {image:image_path}"
    )
    return _filter_records(df, prompt, id_columns=("car_id",))


def _build_animals_q1() -> Tuple[List[BargainRecord], List[Tuple[Any, ...]]]:
    data_dir = os.path.join(SEMBENCH_ROOT, "animals/data/sf_1600")
    df = pd.read_csv(os.path.join(data_dir, "image_data.csv"))
    prompt = "Does the image contain a zebra? Image: {image:ImagePath}"
    return _filter_records(df, prompt, id_columns=("ImagePath",))


# --- LRobench match queries ----------------------------------------------
def _lro_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(LROBENCH_DATABASES_DIR, path))


def _build_match102() -> Tuple[List[BargainRecord], List[Tuple[Any, ...]]]:
    customers = _lro_csv("car_retails/customers.csv")
    customers = customers[customers["creditLimit"] > 100000]
    left = customers.rename(columns={
        "customerNumber": "left_customerNumber", "city": "left_city",
    })[["left_customerNumber", "left_city"]]
    offices = _lro_csv("car_retails/offices.csv")
    right = offices.rename(columns={"territory": "right_territory"})[["right_territory"]]
    right = right.drop_duplicates(subset=["right_territory"])
    prompt = (
        "Determine whether the customer's city is located within the "
        "office's sales territory.\n\n"
        "Customer city: {left_city}\n"
        "Sales territory: {right_territory}"
    )
    return _join_records(
        left, right, prompt,
        left_id_cols=("left_customerNumber",), right_id_cols=("right_territory",),
    )


def _build_match104() -> Tuple[List[BargainRecord], List[Tuple[Any, ...]]]:
    comments = _lro_csv("codebase_community/comments.csv")
    comments = comments.sort_values("Id").head(10)
    left = comments.rename(columns={"Id": "left_Id", "Text": "left_Text"})[["left_Id", "left_Text"]]
    tags = _lro_csv("codebase_community/tags.csv")
    tags = tags.sort_values("Count", ascending=False).head(10)
    right = tags.rename(columns={"Id": "right_Id", "TagName": "right_TagName"})[["right_Id", "right_TagName"]]
    prompt = (
        "Determine whether the comment's content is directly related to the "
        "tag's topic.\n\n"
        "Comment: {left_Text}\n"
        "Tag: {right_TagName}"
    )
    return _join_records(
        left, right, prompt,
        left_id_cols=("left_Id",), right_id_cols=("right_Id",),
    )


def _build_match105() -> Tuple[List[BargainRecord], List[Tuple[Any, ...]]]:
    customer = _lro_csv("shipping/customer.csv")
    customer = customer[customer["cust_type"] == "wholesaler"]
    left = customer.rename(columns={
        "cust_id": "left_cust_id", "state": "left_state",
    })[["left_cust_id", "left_state"]]
    city = _lro_csv("shipping/city.csv")
    city = city[city["population"] > 600000]
    right = city.rename(columns={"state": "right_state"})[["right_state"]]
    right = right.drop_duplicates(subset=["right_state"])
    prompt = (
        "Determine whether the two-letter abbreviated US state code refers to "
        "exactly the same state as the full state name.\n\n"
        "Abbreviated state code: {left_state}\n"
        "Full state name: {right_state}"
    )
    return _join_records(
        left, right, prompt,
        left_id_cols=("left_cust_id",), right_id_cols=("right_state",),
    )


def _build_match107() -> Tuple[List[BargainRecord], List[Tuple[Any, ...]]]:
    team = _lro_csv("european_football_2/Team.csv")
    team = team[team["team_long_name"].str.startswith("P", na=False)]
    left = team.rename(columns={
        "id": "left_id", "team_long_name": "left_team_long_name",
    })[["left_id", "left_team_long_name"]]
    country = _lro_csv("european_football_2/Country.csv")
    right = country.rename(columns={"id": "right_id", "name": "right_name"})[["right_id", "right_name"]]
    prompt = (
        "Determine whether the football team belongs to (plays its domestic "
        "league in) the listed country.\n\n"
        "Team: {left_team_long_name}\n"
        "Country: {right_name}"
    )
    return _join_records(
        left, right, prompt,
        left_id_cols=("left_id",), right_id_cols=("right_id",),
    )


# ---------------------------------------------------------------------------
# Match query ground-truth synthesis (verbatim from the LLMSQL grid).
# Stored as one CSV per query under <results_dir>/<query>_gt.csv so the
# scorer can read it back uniformly.
# ---------------------------------------------------------------------------
_MATCH_GT: Dict[str, List[Tuple[Any, Any]]] = {
    "match102": [
        (114, 'APAC'), (119, 'EMEA'), (141, 'EMEA'), (146, 'EMEA'),
        (148, 'APAC'), (157, 'NA'),   (187, 'EMEA'), (227, 'EMEA'),
        (249, 'EMEA'), (259, 'EMEA'), (276, 'APAC'), (278, 'EMEA'),
        (286, 'NA'),   (298, 'EMEA'), (319, 'NA'),   (363, 'NA'),
        (386, 'EMEA'), (448, 'EMEA'), (458, 'EMEA'), (496, 'APAC'),
    ],
    "match104": [(2, 41), (4, 9), (9, 41), (11, 41)],
    "match105": [
        (381, 'Florida'),     (615, 'Florida'),     (618, 'Florida'),
        (1575, 'California'), (1685, 'California'), (1769, 'Texas'),
        (2001, 'California'), (2799, 'Texas'),      (3288, 'Texas'),
        (3660, 'Maryland'),   (4353, 'Texas'),      (4869, 'Arizona'),
    ],
    "match107": [
        (3476, 1729),   (9548, 4769),   (20532, 10257), (21292, 10257),
        (23523, 10257), (26556, 13274), (29000, 13274), (31444, 15722),
        (31445, 15722), (31448, 15722), (31457, 15722), (32891, 15722),
        (33377, 15722), (36248, 17642), (41673, 19694),
    ],
}


def write_match_gt(query_tag: str, results_dir: str) -> str:
    os.makedirs(results_dir, exist_ok=True)
    gt_path = os.path.join(results_dir, f"{query_tag}_gt.csv")
    pairs = _MATCH_GT[query_tag]
    pd.DataFrame(
        {"id": [f"{a}-{b}" for a, b in pairs]}
    ).to_csv(gt_path, index=False)
    return gt_path


# ---------------------------------------------------------------------------
# Scorer adapters: BARGAIN returns a positive index list; we map that
# back into the (id_components) tuples and then call the pure F1 scorer
# from ``scorers.py``.
# ---------------------------------------------------------------------------
def _hyphen_id_score(scorer):
    """Wrap a single-id-column scorer for queries whose key is a single value."""
    def _inner(predicted_keys, gt_path):
        return scorer((str(k[0]) for k in predicted_keys), gt_path)
    return _inner


def _join_pair_id_score(scorer):
    """Wrap for join queries that compare ``"left-right"`` strings."""
    def _inner(predicted_keys, gt_path):
        ids = ("-".join(str(v) for v in k) for k in predicted_keys)
        return scorer(ids, gt_path)
    return _inner


def _topk_join_score(k: int):
    def _inner(predicted_keys, gt_path):
        ids = ["-".join(str(v) for v in key) for key in predicted_keys]
        return scorers.score_id_set_topk(ids, gt_path, k=k)
    return _inner


def _mmqa_q2a_score(predicted_keys, _gt_path):
    return scorers.score_mmqa_q2a(predicted_keys)


def _mmqa_q7_score(predicted_keys, _gt_path):
    return scorers.score_mmqa_q7(predicted_keys)


def _animals_q1_score(predicted_keys, gt_path):
    return scorers.score_animals_q1((k[0] for k in predicted_keys), gt_path)


def _cars_q3_score(predicted_keys, gt_path):
    return scorers.score_cars_q3((k[0] for k in predicted_keys), gt_path)


def _cars_q4_score(predicted_keys, gt_path):
    return scorers.score_cars_q4((k[0] for k in predicted_keys), gt_path)


def _cars_q8_score(predicted_keys, gt_path):
    return scorers.score_cars_q8((k[0] for k in predicted_keys), gt_path)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def build_query_registry(results_dir: str) -> Dict[str, Query]:
    return {
        "ecomm_q7": Query(
            tag="ecomm_q7", kind="join", modality="text",
            prompt="see _build_ecomm_q7",
            gt_path=os.path.join(SEMBENCH_ROOT, "ecomm/raw_results/ground_truth/Q7.csv"),
            score_fn=_topk_join_score(int(os.environ.get("ECOMM_Q7_TOPK", "0"))),
            build_fn=_build_ecomm_q7,
            id_components=("left_id", "right_id"),
        ),
        "ecomm_q8": Query(
            tag="ecomm_q8", kind="join", modality="multimodal",
            prompt="see _build_ecomm_q8",
            gt_path=os.path.join(SEMBENCH_ROOT, "ecomm/raw_results/ground_truth/Q8.csv"),
            score_fn=_join_pair_id_score(scorers.score_id_set),
            build_fn=_build_ecomm_q8,
            id_components=("style_id", "image_id"),
        ),
        "ecomm_q9": Query(
            tag="ecomm_q9", kind="join", modality="multimodal",
            prompt="see _build_ecomm_q9",
            gt_path=os.path.join(SEMBENCH_ROOT, "ecomm/raw_results/ground_truth/Q9.csv"),
            score_fn=_join_pair_id_score(scorers.score_id_set),
            build_fn=_build_ecomm_q9,
            id_components=("left_id", "right_id"),
        ),
        "ecomm_q13": Query(
            tag="ecomm_q13", kind="filter", modality="multimodal",
            prompt="see _build_ecomm_q13",
            gt_path=os.path.join(SEMBENCH_ROOT, "ecomm/raw_results/ground_truth/Q13.csv"),
            score_fn=_hyphen_id_score(scorers.score_id_set),
            build_fn=_build_ecomm_q13,
            id_components=("id",),
        ),
        "mmqa_q2a": Query(
            tag="mmqa_q2a", kind="join", modality="multimodal",
            prompt="see _build_mmqa_q2a",
            gt_path=None,
            score_fn=_mmqa_q2a_score,
            build_fn=_build_mmqa_q2a,
            id_components=("ID", "image_filename"),
        ),
        "mmqa_q7": Query(
            tag="mmqa_q7", kind="join", modality="multimodal",
            prompt="see _build_mmqa_q7",
            gt_path=None,
            score_fn=_mmqa_q7_score,
            build_fn=_build_mmqa_q7,
            id_components=("Airlines", "image_filename"),
        ),
        "cars_q3": Query(
            tag="cars_q3", kind="filter", modality="multimodal",
            prompt="see _build_cars_q3",
            gt_path=os.path.join(SEMBENCH_ROOT, "cars/raw_results/ground_truth/Q3.csv"),
            score_fn=_cars_q3_score,
            build_fn=_build_cars_q3,
            id_components=("vin",),
        ),
        "cars_q4": Query(
            tag="cars_q4", kind="filter", modality="text",
            prompt="see _build_cars_q4",
            gt_path=os.path.join(SEMBENCH_ROOT, "cars/raw_results/ground_truth/Q4.csv"),
            score_fn=_cars_q4_score,
            build_fn=_build_cars_q4,
            id_components=("car_id",),
        ),
        "cars_q8": Query(
            tag="cars_q8", kind="filter", modality="multimodal",
            prompt="see _build_cars_q8",
            gt_path=os.path.join(SEMBENCH_ROOT, "cars/raw_results/ground_truth/Q8.csv"),
            score_fn=_cars_q8_score,
            build_fn=_build_cars_q8,
            id_components=("car_id",),
        ),
        "animals_q1": Query(
            tag="animals_q1", kind="filter", modality="multimodal",
            prompt="see _build_animals_q1",
            gt_path=os.path.join(SEMBENCH_ROOT, "animals/raw_results/ground_truth/Q1.csv"),
            score_fn=_animals_q1_score,
            build_fn=_build_animals_q1,
            id_components=("ImagePath",),
        ),
        "match102": Query(
            tag="match102", kind="join", modality="text",
            prompt="see _build_match102",
            gt_path=write_match_gt("match102", results_dir),
            score_fn=_join_pair_id_score(scorers.score_id_set),
            build_fn=_build_match102,
            id_components=("left_customerNumber", "right_territory"),
        ),
        "match104": Query(
            tag="match104", kind="join", modality="text",
            prompt="see _build_match104",
            gt_path=write_match_gt("match104", results_dir),
            score_fn=_join_pair_id_score(scorers.score_id_set),
            build_fn=_build_match104,
            id_components=("left_Id", "right_Id"),
        ),
        "match105": Query(
            tag="match105", kind="join", modality="text",
            prompt="see _build_match105",
            gt_path=write_match_gt("match105", results_dir),
            score_fn=_join_pair_id_score(scorers.score_id_set),
            build_fn=_build_match105,
            id_components=("left_cust_id", "right_state"),
        ),
        "match107": Query(
            tag="match107", kind="join", modality="text",
            prompt="see _build_match107",
            gt_path=write_match_gt("match107", results_dir),
            score_fn=_join_pair_id_score(scorers.score_id_set),
            build_fn=_build_match107,
            id_components=("left_id", "right_id"),
        ),
    }


# ---------------------------------------------------------------------------
# Default subsets exposed via the CLI
# ---------------------------------------------------------------------------
ALL_TAGS: Tuple[str, ...] = (
    "ecomm_q7", "ecomm_q8", "ecomm_q9", "ecomm_q13",
    "mmqa_q2a", "mmqa_q7",
    "cars_q3", "cars_q4", "cars_q8",
    "animals_q1",
    "match102", "match104", "match105", "match107",
)

GROUPS: Dict[str, Tuple[str, ...]] = {
    "ecomm": ("ecomm_q7", "ecomm_q8", "ecomm_q9", "ecomm_q13"),
    "mmqa": ("mmqa_q2a", "mmqa_q7"),
    "cars": ("cars_q3", "cars_q4", "cars_q8"),
    "animals": ("animals_q1",),
    "match": ("match102", "match104", "match105", "match107"),
    "all": ALL_TAGS,
}
