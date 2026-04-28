"""vLLM-backed BARGAIN Proxy and Oracle.

The galaxy-grid benchmarks hit a Qwen3-VL vLLM server (the same
deployment LLMSQL uses) over its OpenAI-compatible HTTP API. Each
``data_record`` passed into BARGAIN is a :class:`BargainRecord` dict
holding the rendered text prompt and any image paths the prompt
references; this module turns one of those records into a chat
completion request and parses the True/False answer (plus its
log-probability for the proxy).

The proxy and oracle are intentionally cheap to construct — the
expensive object is the :class:`openai.OpenAI` HTTP client, which is
created lazily and shared across calls.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from BARGAIN.models.AbstractModels import Oracle, Proxy

logger = logging.getLogger(__name__)


_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff")


@dataclass
class BargainRecord:
    """A single record handed to BARGAIN as ``data_records[i]``.

    ``prompt`` is the fully-rendered user message text (with all
    ``{col}`` placeholders substituted). ``images`` are absolute paths
    to any image columns; they are attached to the chat request as
    OpenAI-format ``image_url`` content blocks.
    ``id`` is an opaque identifier used by the scorer to map the
    returned positive indexes back to ground-truth rows (e.g. a
    composite ``"left_id-right_id"`` for joins, a single ``"car_id"``
    for filters).
    """

    id: str
    prompt: str
    images: Tuple[str, ...] = field(default_factory=tuple)
    extras: Dict[str, Any] = field(default_factory=dict)


def _normalize_image_url(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith(("file://", "http://", "https://", "data:")):
        return raw
    return f"file://{raw}"


def _is_image_value(value: str) -> bool:
    lower = value.lower().strip()
    if any(lower.endswith(ext) for ext in _IMAGE_EXTENSIONS):
        return True
    return lower.startswith(("file://", "http://", "https://", "data:image/"))


def _build_user_content(record: BargainRecord) -> Any:
    """Return the ``messages[user]['content']`` payload for a record.

    For text-only records this is just the raw prompt string; for
    multimodal records it is the OpenAI vision content list mixing one
    text block with one ``image_url`` block per attached image.
    """
    if not record.images:
        return record.prompt
    blocks: List[Dict[str, Any]] = [{"type": "text", "text": record.prompt}]
    for img in record.images:
        blocks.append({
            "type": "image_url",
            "image_url": {"url": _normalize_image_url(img)},
        })
    return blocks


def _parse_bool(content: str) -> bool:
    if content is None:
        return False
    text = content.strip().lower()
    if "true" in text and "false" not in text:
        return True
    if "false" in text and "true" not in text:
        return False
    return text.startswith("t")


def _extract_bool_logprob(logprobs_obj) -> Tuple[bool, float]:
    """Recover (label, calibrated_prob) from an OpenAI logprobs payload.

    Mirrors the scoring path used by ``BARGAIN.models.GPTModels`` so the
    proxy score is always in [0, 1] and can be fed into BARGAIN's
    cascade-threshold solver.
    """
    if logprobs_obj is None:
        return False, 0.0
    try:
        first_token = logprobs_obj.content[0]
    except (AttributeError, IndexError):
        return False, 0.0

    true_prob = 0.0
    false_prob = 0.0
    for tlp in getattr(first_token, "top_logprobs", []) or []:
        token = (tlp.token or "").strip().lower()
        if token in ("true", "yes", "1"):
            true_prob = max(true_prob, float(np.exp(tlp.logprob)))
        elif token in ("false", "no", "0"):
            false_prob = max(false_prob, float(np.exp(tlp.logprob)))

    if true_prob == 0.0 and false_prob == 0.0:
        # Top-K didn't include either alternative — fall back to the
        # parsed text label with a neutral confidence so the score
        # column still has a valid number.
        token_text = (getattr(first_token, "token", "") or "").strip().lower()
        if token_text.startswith("t"):
            return True, 0.5
        return False, 0.5

    norm = true_prob + false_prob
    true_prob /= norm
    false_prob /= norm
    if true_prob >= false_prob:
        return True, float(true_prob)
    return False, float(false_prob)


def _build_openai_client(base_url: str, timeout: float = 600.0):
    from openai import OpenAI

    return OpenAI(
        base_url=base_url.rstrip("/"),
        api_key=os.environ.get("VLLM_API_KEY", "dummy"),
        timeout=timeout,
    )


def _record_message(record: BargainRecord, system_prompt: str) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _build_user_content(record)},
    ]


_DEFAULT_VLLM_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
_DEFAULT_PROXY_MODEL = os.environ.get(
    "BARGAIN_PROXY_MODEL", "Qwen/Qwen3-VL-2B-Instruct",
)
_DEFAULT_ORACLE_MODEL = os.environ.get(
    "BARGAIN_ORACLE_MODEL", "Qwen/Qwen3-VL-30B-A3B-Instruct",
)
_DEFAULT_FILTER_SYSTEM_PROMPT = (
    "You are a helpful data analyst. You will receive data and a query. "
    "Answer ONLY 'True' or 'False'."
)
_DEFAULT_JOIN_SYSTEM_PROMPT = (
    "You are a helpful data analyst. You will receive data from two "
    "tables and a query. Answer ONLY 'True' or 'False'."
)


class VLLMProxy(Proxy):
    """Cheap proxy: small Qwen3-VL on the same vLLM server."""

    def __init__(
        self,
        model: str = _DEFAULT_PROXY_MODEL,
        base_url: str = _DEFAULT_VLLM_URL,
        system_prompt: str = _DEFAULT_FILTER_SYSTEM_PROMPT,
        max_tokens: int = 4,
        max_workers: int = 32,
        verbose: bool = True,
    ) -> None:
        super().__init__(verbose=verbose)
        self.model = model
        self.base_url = base_url
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.max_workers = max_workers
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = _build_openai_client(self.base_url)
        return self._client

    def _call_one(self, record: BargainRecord) -> Tuple[bool, float]:
        messages = _record_message(record, self.system_prompt)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            max_tokens=self.max_tokens,
            logprobs=True,
            top_logprobs=10,
            seed=0,
        )
        choice = response.choices[0]
        return _extract_bool_logprob(choice.logprobs)

    def proxy_func(self, data_record: BargainRecord) -> Tuple[bool, float]:
        try:
            return self._call_one(data_record)
        except Exception as exc:  # noqa: BLE001 — surface and degrade
            logger.warning("Proxy call failed for %s: %s", data_record.id, exc)
            return False, 0.0

    # The default ``get_preds_and_scores`` issues serial calls via
    # ``proxy_func``, which is intolerably slow for the join queries
    # (millions of pairs). Override to fan out across a thread pool
    # while preserving the per-record cache the abstract base relies on.
    def get_preds_and_scores(
        self,
        indxs: Sequence[int],
        data_records: Sequence[BargainRecord],
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(indxs)
        preds = np.empty(n, dtype=bool)
        scores = np.empty(n, dtype=float)
        pending: Dict[int, int] = {}  # future_id -> position

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = []
            for pos, key in enumerate(indxs):
                if key in self.preds_dict:
                    pred, score = self.preds_dict[key]
                    preds[pos] = pred
                    scores[pos] = score
                    continue
                fut = pool.submit(self._call_one, data_records[pos])
                pending[id(fut)] = pos
                futures.append((fut, pos, key))

            iterator = as_completed(f for f, _, _ in futures)
            if self.verbose and len(futures) > 20:
                iterator = tqdm(iterator, total=len(futures), desc="proxy")
            done_lookup = {id(f): (pos, key) for f, pos, key in futures}
            for fut in iterator:
                pos, key = done_lookup[id(fut)]
                try:
                    pred, score = fut.result()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Proxy batch call failed: %s", exc)
                    pred, score = False, 0.0
                self.preds_dict[key] = (pred, score)
                preds[pos] = pred
                scores[pos] = score

        return preds, scores


class VLLMOracle(Oracle):
    """Expensive oracle: large Qwen3-VL on the same vLLM server.

    BARGAIN expects the oracle to return the *correct* label for each
    record. For binary semantic filters/joins that's a True/False
    decision, so :meth:`oracle_func` simply parses the model's reply
    and reports whether it agrees with ``proxy_output``.
    """

    def __init__(
        self,
        model: str = _DEFAULT_ORACLE_MODEL,
        base_url: str = _DEFAULT_VLLM_URL,
        system_prompt: str = _DEFAULT_FILTER_SYSTEM_PROMPT,
        max_tokens: int = 4,
        max_workers: int = 16,
        verbose: bool = True,
    ) -> None:
        super().__init__(verbose=verbose)
        self.model = model
        self.base_url = base_url
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.max_workers = max_workers
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = _build_openai_client(self.base_url)
        return self._client

    def _call_one(self, record: BargainRecord) -> bool:
        messages = _record_message(record, self.system_prompt)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            max_tokens=self.max_tokens,
            seed=0,
        )
        return _parse_bool(response.choices[0].message.content)

    def oracle_func(self, data_record: BargainRecord, proxy_output: Any) -> Tuple[bool, bool]:
        try:
            oracle_output = self._call_one(data_record)
        except Exception as exc:  # noqa: BLE001 — surface and degrade
            logger.warning("Oracle call failed for %s: %s", data_record.id, exc)
            oracle_output = False
        return (oracle_output == bool(proxy_output)), oracle_output

    # Override ``get_pred`` to parallelise the bulk-labelling calls
    # BARGAIN_P/BARGAIN_R make at the end of ``process``.
    def get_pred(
        self,
        data_records: Sequence[BargainRecord],
        indxs: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        n = len(data_records)
        out = np.empty(n, dtype=bool)
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {}
            for pos, record in enumerate(data_records):
                key = None if indxs is None else indxs[pos]
                if key is not None and key in self.preds_dict:
                    out[pos] = self.preds_dict[key]
                    continue
                fut = pool.submit(self._call_one, record)
                futures[fut] = (pos, key)

            iterator = as_completed(futures)
            if self.verbose and len(futures) > 20:
                iterator = tqdm(iterator, total=len(futures), desc="oracle")
            for fut in iterator:
                pos, key = futures[fut]
                try:
                    label = fut.result()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Oracle batch call failed: %s", exc)
                    label = False
                if key is not None:
                    self.preds_dict[key] = label
                out[pos] = label
        return out
