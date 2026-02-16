"""Helpers for reading backend store snapshots from a compiled graph."""

from __future__ import annotations

import inspect
from typing import Any


def snapshot_backend_store(
    graph: Any,
    *,
    namespace_prefix: tuple[str, ...] | None = None,
    max_namespaces: int = 20,
    max_items_per_namespace: int = 20,
) -> tuple[dict[str, dict[str, Any]], str, str | None]:
    """Return a normalized backend-store snapshot.

    Returns:
        (items, source, error)
        - items: namespace-string -> {key: value}
        - source: "backend", "backend-legacy", "none", "unsupported", "error"
        - error: optional error message
    """
    store = getattr(graph, "store", None)
    if store is None:
        return {}, "none", None

    try:
        if _has_callable(store, "list_namespaces") and _has_callable(store, "search"):
            return (
                _snapshot_via_base_store_api(
                    store,
                    namespace_prefix=namespace_prefix,
                    max_namespaces=max_namespaces,
                    max_items_per_namespace=max_items_per_namespace,
                ),
                "backend",
                None,
            )

        legacy = _snapshot_legacy_store(
            store,
            namespace_prefix=namespace_prefix,
            max_namespaces=max_namespaces,
        )
        if legacy is not None:
            return legacy, "backend-legacy", None

        return {}, "unsupported", "Store does not expose a recognized read API."
    except Exception as e:
        return {}, "error", str(e)


def _snapshot_via_base_store_api(
    store: Any,
    *,
    namespace_prefix: tuple[str, ...] | None,
    max_namespaces: int,
    max_items_per_namespace: int,
) -> dict[str, dict[str, Any]]:
    list_kwargs: dict[str, Any] = {"limit": max_namespaces, "offset": 0}
    if namespace_prefix is not None:
        list_kwargs["prefix"] = namespace_prefix

    namespaces_raw = _call_with_supported_kwargs(store.list_namespaces, **list_kwargs)
    if not isinstance(namespaces_raw, list):
        return {}

    normalized: dict[str, dict[str, Any]] = {}
    for ns_raw in namespaces_raw[:max_namespaces]:
        namespace = _normalize_namespace(ns_raw)
        if namespace is None:
            continue
        if namespace_prefix and not namespace[: len(namespace_prefix)] == namespace_prefix:
            continue

        search_results = _call_with_supported_kwargs(
            store.search,
            namespace,
            limit=max_items_per_namespace,
            offset=0,
            refresh_ttl=False,
        )
        if not isinstance(search_results, list):
            continue

        entries: dict[str, Any] = {}
        for item in search_results[:max_items_per_namespace]:
            key, value = _extract_item(item)
            entries[key] = value
        normalized[_namespace_key(namespace)] = entries

    return normalized


def _snapshot_legacy_store(
    store: Any,
    *,
    namespace_prefix: tuple[str, ...] | None,
    max_namespaces: int,
) -> dict[str, dict[str, Any]] | None:
    if _has_callable(store, "list"):
        prefix = _namespace_key(namespace_prefix) if namespace_prefix else None
        if prefix is not None:
            data = store.list([prefix])
        else:
            # Some legacy stores require explicit prefixes.
            data_attr = getattr(store, "data", None)
            if isinstance(data_attr, dict):
                data = store.list([str(k) for k in list(data_attr.keys())[:max_namespaces]])
            else:
                data = {}
        if isinstance(data, dict):
            return _normalize_legacy_mapping(data, max_namespaces=max_namespaces)

    data_attr = getattr(store, "data", None)
    if isinstance(data_attr, dict):
        return _normalize_legacy_mapping(data_attr, max_namespaces=max_namespaces)

    return None


def _normalize_legacy_mapping(
    mapping: dict[Any, Any], *, max_namespaces: int
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for ns, entries in list(mapping.items())[:max_namespaces]:
        if not isinstance(entries, dict):
            continue
        out[str(ns)] = {str(k): v for k, v in entries.items()}
    return out


def _extract_item(item: Any) -> tuple[str, Any]:
    if hasattr(item, "key"):
        key = getattr(item, "key")
        value = getattr(item, "value", item)
        return str(key), value

    if isinstance(item, dict):
        key = item.get("key")
        if key is not None:
            return str(key), item.get("value", item)

    return repr(item), item


def _namespace_key(namespace: tuple[str, ...] | None) -> str:
    if not namespace:
        return "<root>"
    return "/".join(namespace)


def _normalize_namespace(value: Any) -> tuple[str, ...] | None:
    if isinstance(value, tuple):
        return tuple(str(part) for part in value)
    if isinstance(value, list):
        return tuple(str(part) for part in value)
    return None


def _has_callable(obj: Any, name: str) -> bool:
    return callable(getattr(obj, name, None))


def _call_with_supported_kwargs(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Call function with only kwargs it supports."""
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return func(*args)

    supported = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return func(*args, **supported)
