"""BARGAIN benchmark runners for the LLMSQL galaxy grid queries.

This subpackage adapts a subset of the queries from the LLMSQL
``run_direct_galaxy_grid.py`` harness so they can be executed through
the BARGAIN proxy/oracle cascade instead of LLMSQL's
sample-then-propagate pipeline.

Each query produces a list of ``data records`` (per-row for sem_filter,
per-pair for sem_join). BARGAIN's :class:`BARGAIN_P` then chooses which
records to send to the oracle (Qwen3-VL-30B) and which to trust the
proxy (Qwen3-VL-2B by default) on, while guaranteeing a target
precision on the returned positive set.

Modules
-------
queries
    Per-query data loaders, prompt templates, ground-truth paths, and
    scorer factories.
vllm_models
    :class:`VLLMProxy` and :class:`VLLMOracle` — concrete BARGAIN
    Proxy/Oracle implementations that talk to a running vLLM server
    over the OpenAI-compatible HTTP API. Both text-only and multimodal
    (``{image:col}``) prompts are handled.
scorers
    F1 / aggregation scorers ported from the LLMSQL grid so the BARGAIN
    benchmark numbers stay directly comparable to the LLMSQL
    direct/galaxy numbers.
"""
