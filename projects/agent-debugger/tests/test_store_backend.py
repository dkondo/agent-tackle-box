"""Tests for backend store snapshot helpers."""

from __future__ import annotations

from agent_debugger.store_backend import snapshot_backend_store


def test_snapshot_backend_store_returns_none_source_without_store():
    class _Graph:
        pass

    items, source, error = snapshot_backend_store(_Graph())
    assert items == {}
    assert source == "none"
    assert error is None


def test_snapshot_backend_store_reads_base_store_shape():
    class _Item:
        def __init__(self, key, value):
            self.key = key
            self.value = value

    class _Store:
        def list_namespaces(self, **kwargs):
            return [("memories", "u1")]

        def search(self, namespace_prefix, **kwargs):
            if namespace_prefix == ("memories", "u1"):
                return [_Item("k1", {"v": 1})]
            return []

    class _Graph:
        store = _Store()

    items, source, error = snapshot_backend_store(_Graph())
    assert source == "backend"
    assert error is None
    assert items == {"memories/u1": {"k1": {"v": 1}}}


def test_snapshot_backend_store_reads_legacy_list_shape():
    class _Store:
        def __init__(self):
            self.data = {"memories/u1": {"k1": {"v": 1}}}

        def list(self, prefixes):
            return {prefix: self.data.get(prefix, {}) for prefix in prefixes}

    class _Graph:
        store = _Store()

    items, source, error = snapshot_backend_store(_Graph())
    assert source == "backend-legacy"
    assert error is None
    assert items == {"memories/u1": {"k1": {"v": 1}}}
