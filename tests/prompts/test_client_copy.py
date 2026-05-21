"""Regression: ``LLMClient.copy()`` must share its ``_call_logs`` list with
the parent so per-call appends from concurrent ``copy()`` instances roll up
into the parent's ledger.

Two surfaces depend on this:
  * ``prompts/client.py`` runs an in-band budget guard
    (``sum(log.cost_utilization for log in self._call_logs) > 1.0``) on every
    call; a fresh-list copy lets each parallel call see only its own cost
    and bypass ``max_cost`` by a factor equal to the parallel fan-out.
  * Callers that ``await asyncio.gather(*(client.copy()(p) for p in …))``
    and then read ``client.call_logs`` expect to see all the per-call
    entries; a fresh-list copy orphans them on garbage-collected views.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_ROOT = Path(__file__).resolve().parents[2]


def _load_chains_client() -> ModuleType:
    """Load ``problems/chains/client.py`` standalone.

    The real module imports ``openai`` and ``gigaevo.infra.aiohttp_factory``;
    both resolve in the test venv, so we just load by path under a
    distinct name and avoid the ``tests/problems/__init__.py`` shadow.
    """
    name = "_chains_client_under_test"
    if name in sys.modules:
        return sys.modules[name]
    path = _ROOT / "problems" / "chains" / "client.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_mod = _load_chains_client()
LLMClient = _mod.LLMClient
CallLog = _mod.CallLog


def _bare_client() -> "LLMClient":
    """Construct an ``LLMClient`` without touching the network layer."""
    c = LLMClient.__new__(LLMClient)
    c.model = "openai/gpt-4o-mini"
    c.max_cost = 10.0
    c.model_pricing = {"prompt": 0.15, "completion": 0.60}
    c.generation_kwargs = {"max_tokens": 32768}
    c.client = None
    c._call_logs = []
    return c


def test_copy_shares_call_logs_with_parent() -> None:
    parent = _bare_client()
    child = parent.copy()
    assert child._call_logs is parent._call_logs, (
        "copy() must share the call_logs list reference; a fresh list "
        "orphans per-call appends and silently defeats cost tracking."
    )


def test_copy_append_visible_on_parent() -> None:
    parent = _bare_client()
    child_a = parent.copy()
    child_b = parent.copy()
    child_a._call_logs.append(
        CallLog(prompt_tokens=10, completion_tokens=20, cost=0.5, cost_utilization=0.05)
    )
    child_b._call_logs.append(
        CallLog(prompt_tokens=12, completion_tokens=25, cost=0.6, cost_utilization=0.06)
    )
    assert len(parent.call_logs) == 2
    assert parent.call_logs[0].cost == 0.5
    assert parent.call_logs[1].cost == 0.6


def test_copy_chain_preserves_sharing() -> None:
    """Copy-of-a-copy still pins to the original parent's ledger."""
    parent = _bare_client()
    grandchild = parent.copy().copy()
    grandchild._call_logs.append(
        CallLog(prompt_tokens=1, completion_tokens=2, cost=0.01, cost_utilization=0.001)
    )
    assert len(parent.call_logs) == 1


def test_clear_logs_preserves_sharing_with_copies() -> None:
    """``clear_logs`` must mutate the shared list in place.

    Rebinding ``self._call_logs = []`` on the parent would silently detach
    any outstanding ``copy()`` from the budget ledger: post-clear appends
    from the copy would accumulate on a now-orphaned list and ``max_cost``
    would observe zero new cost. The implementation mutates via
    ``list.clear()``.
    """
    parent = _bare_client()
    child = parent.copy()
    child._call_logs.append(
        CallLog(prompt_tokens=1, completion_tokens=1, cost=0.5, cost_utilization=0.05)
    )
    assert len(parent.call_logs) == 1

    parent.clear_logs()

    assert child._call_logs is parent._call_logs, (
        "clear_logs() rebound the list reference; existing copies are now "
        "orphaned from the budget ledger."
    )
    child._call_logs.append(
        CallLog(prompt_tokens=2, completion_tokens=2, cost=0.9, cost_utilization=0.09)
    )
    assert len(parent.call_logs) == 1
    assert parent.call_logs[0].cost == 0.9
