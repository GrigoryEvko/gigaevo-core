"""Regression: ``tools.no_proxy`` must not crash on machines that have
not provisioned ``experiments/infrastructure.yaml``.

The function is called at module import time by
``problems/chains/*/shared_config.py``; a ``FileNotFoundError`` there
killed every ``shared_config`` import (and therefore every chains-
problem entrypoint) on fresh checkouts.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _load_no_proxy_with_missing_yaml(missing_path: Path) -> ModuleType:
    """Re-import ``tools.no_proxy`` against a yaml path that does not
    exist. Returns the freshly loaded module."""
    name = "_tools_no_proxy_under_test"
    sys.modules.pop(name, None)
    src = _ROOT / "tools" / "no_proxy.py"
    spec = importlib.util.spec_from_file_location(name, src)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    module._INFRA_YAML = missing_path  # type: ignore[attr-defined]
    sys.modules[name] = module
    spec.loader.exec_module(module)
    # exec_module ran before our patch took effect; reset and re-derive.
    module._INFRA_YAML = missing_path  # type: ignore[attr-defined]
    return module


def test_get_no_proxy_hosts_returns_empty_when_yaml_missing(tmp_path) -> None:
    mod = _load_no_proxy_with_missing_yaml(tmp_path / "does_not_exist.yaml")
    assert mod.get_no_proxy_hosts() == []


def test_get_no_proxy_string_returns_empty_when_yaml_missing(tmp_path) -> None:
    mod = _load_no_proxy_with_missing_yaml(tmp_path / "does_not_exist.yaml")
    assert mod.get_no_proxy_string() == ""


def test_ensure_no_proxy_is_silent_when_yaml_missing(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of the fix: a missing infrastructure.yaml must
    not raise. ``ensure_no_proxy`` should still scrub HTTP(S)_PROXY
    env vars so the proxy-bypass contract holds in the no-config case."""
    monkeypatch.setenv("HTTP_PROXY", "http://stale.proxy:3128")
    monkeypatch.setenv("HTTPS_PROXY", "http://stale.proxy:3128")
    mod = _load_no_proxy_with_missing_yaml(tmp_path / "does_not_exist.yaml")
    mod.ensure_no_proxy()
    import os

    assert os.environ.get("HTTP_PROXY") is None
    assert os.environ.get("HTTPS_PROXY") is None
