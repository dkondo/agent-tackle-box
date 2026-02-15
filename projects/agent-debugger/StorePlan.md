# Store Backend Plan

## Goal
Show true backend store data in adb's Store panel (LangGraph `BaseStore`), and never infer Store panel content from graph state when backend store is absent.

## Scope
- First-class backend-store reads via `list_namespaces` + `search`.
- Legacy store read compatibility for recognizable memory-store shapes.
- Source/error metadata surfaced in UI (`backend`, `backend-legacy`, `none`, `unsupported`, `error`).
- Optional renderer interfaces for Store and State customization.

## State Pane Clarification
- Default `State` pane already shows all top-level state fields.
- `messages` is summarized/collapsible for readability.
- Any custom state keys are shown by default unless a custom renderer overrides the view.

## Extension Interfaces
- `StoreRenderer`: optional custom rendering for backend store snapshots.
- `StateRenderer`: optional custom rendering for state pane output.
- `MemoryRenderer`: retained for backward compatibility, but only applied for real backend snapshots (`backend` / `backend-legacy`). It is not used as a fallback when backend store is missing.

## Implementation Plan
1. Backend snapshot adapter
- Add `adb/store_backend.py` to normalize backend store reads.
- Read `runner.graph.store` and normalize to `namespace -> {key: value}`.

2. Legacy snapshot adapter
- Detect legacy store shapes (`list(...)` / `data`) and normalize output.
- Mark source as `backend-legacy`.

3. Event payload updates
- Extend `StateUpdateEvent` with `store_items`, `store_source`, `store_error`.
- Emit backend snapshot metadata from runner.

4. Runner integration
- Fetch store snapshots in `AgentRunner` during state updates.
- Include store snapshot in dedupe signature.
- Keep failures non-fatal and surfaced as metadata.

5. Store panel behavior
- Render Store panel from backend snapshot only.
- If backend snapshot is absent/unsupported/error, show empty backend view (no state fallback).
- Display source/error labels in Store panel.

6. Renderer wiring
- Wire `StoreRenderer` through extensions, app, and CLI.
- Wire `StateRenderer` through extensions, app, and CLI.
- Rendering order for Store panel:
  1. `StoreRenderer`
  2. `MemoryRenderer` (backend snapshots only)
  3. default backend store rendering

7. Store snapshot controls
- CLI flags:
  - `--store-prefix`
  - `--store-max-namespaces`
  - `--store-items-per-namespace`
- Manual refresh command:
  - `/store`

8. Tests
- Adapter tests for no-store, base-store, legacy-store.
- Runner tests for store snapshot emission and missing-store metadata.
- App tests for:
  - backend-first Store rendering
  - no graph-state fallback when backend store is missing
  - `StoreRenderer` and `StateRenderer` behavior
- CLI tests for new renderer flags and store snapshot options.

## Acceptance Criteria
1. Store panel displays backend-only store data and can show data not present in graph state.
2. Store panel does not infer content from graph state when backend store is missing.
3. `InMemoryStore`/`MemoryStore` and `PostgresStore`-style `BaseStore` APIs are supported.
4. State pane shows custom state fields by default and can be fully customized via `StateRenderer`.
