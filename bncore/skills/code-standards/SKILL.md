---
name: code-standards
description: Enforce production-quality Python coding standards for the PyBNCore GUI — typing, structure, errors, tests. Always active when writing or reviewing Python code in `pybncore_gui/`.
---

# Always active
This skill applies to every file in `pybncore_gui/` and `tests/gui/`.

# Typing
- Type hints on every function parameter, return, and class attribute.
- Use `from __future__ import annotations` in every module.
- Use `Mapping`/`Sequence` for inputs and `list`/`dict` for outputs when you intend mutation.
- Use `@dataclass(frozen=True, slots=True)` for DTOs.
- Use `typing.Protocol` for service interfaces when more than one implementation may exist.

# Module conventions
- One primary class per file, plus tightly coupled helpers.
- Max ~500 lines per file; split by responsibility.
- Module-level docstring: one line describing purpose.
- Import order: `from __future__` → stdlib → third-party → Qt → local `domain`/`services`/etc.
- Never `from x import *`.

# Naming
- Classes: `PascalCase` — `NodeInspectorViewModel`, not `node_inspector_vm`.
- Functions & variables: `snake_case`.
- Constants: `UPPER_SNAKE`.
- Signals: `past_tense_or_noun` — `node_selected`, `query_completed`.
- Qt slots: prefix `_on_` when they are private handlers: `_on_compile_clicked`.

# Errors
- Domain exceptions live in `domain/errors.py`: `CompileError`, `QueryError`, `EvidenceError`, `IOError`, `CancelledError`.
- Every public service method catches the relevant wrapper exception and raises the matching domain exception with a user-facing message.
- No bare `except:`. No `except Exception:` except in worker barriers (documented in `qt-workers`).
- No silent failures — if something is skipped, log it via `logging.getLogger(__name__).warning(...)`.

# Logging
- One `logging.getLogger(__name__)` per module.
- `DEBUG` for worker progress, `INFO` for user-initiated actions, `WARNING` for recoverable issues, `ERROR` for failures shown to the user.
- A file handler writes to `<user-cache-dir>/pybncore_gui.log` with rotation; a Qt handler pipes ERROR/WARNING into the bottom-panel Logs tab.

# Documentation
- Docstrings on every public class and method. One-sentence summary; add `Args`/`Returns`/`Raises` only when non-obvious.
- No inline comments that restate the code. Only explain *why*.

# Testing
- `tests/gui/services/` → full coverage of every service method using a real `PyBNCoreWrapper` on small fixtures (`tests/gui/fixtures/*.xdsl`).
- `tests/gui/viewmodels/` → unit tests using `pytest-qt`'s `qtbot` and mocked services.
- `tests/gui/workers/` → use `pytest-qt` `waitSignal`.
- Views are smoke-tested only (construct, show, close).
- Target: services ≥ 90 %, viewmodels ≥ 75 %.

# Dependencies
Add to `pyproject.toml` under an optional `gui` extra:
- `PySide6>=6.6`
- `numpy` (already in core)
- `pyqtgraph>=0.13` (bar charts, CDF plots)
- Dev: `pytest`, `pytest-qt`, `pytest-cov`, `ruff`, `mypy`.

# Lint & format
- `ruff` with rules: `E, F, I, N, UP, B, SIM, RUF`.
- `mypy --strict` on `domain/` and `services/`; `--no-strict-optional` on Qt-adjacent code.

# Bans
- No `print()` — use the logger.
- No globals for state. Inject dependencies.
- No magic numbers. Promote to a module-level `UPPER_SNAKE` constant.
- No commented-out code in committed files.
- No TODO without an owner and context: `# TODO(@akram, phase-3): add undo for CPT edits`.
