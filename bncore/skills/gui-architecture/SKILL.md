---
name: gui-architecture
description: Design and enforce modular architecture for the PyBNCore desktop GUI with strict separation between views, viewmodels, services, models, and workers. Use whenever creating, moving, or reviewing any structural piece of the GUI codebase.
---

# When to use
- Creating or moving any module inside `pybncore_gui/`.
- Deciding where new logic belongs (UI vs. viewmodel vs. service).
- Reviewing a PR for architectural drift.

# Target package layout
```
pybncore_gui/
├── app.py                  # QApplication bootstrap, DI wiring, main entry
├── domain/                 # Pure-Python dataclasses (no Qt imports)
│   ├── node.py
│   ├── evidence.py
│   ├── results.py          # PosteriorResult, MAPResult, SensitivityResult, VOIResult, ContinuousPosteriorDTO
│   └── session.py          # ModelSession (owns the current wrapper)
├── services/               # Thin facade over PyBNCoreWrapper (no Qt imports)
│   ├── model_service.py    # authoring: add/remove nodes, edges, CPTs
│   ├── io_service.py       # XDSL/BIF read/write
│   ├── inference_service.py  # compile, query, batch_query, MAP, soft evidence
│   ├── analysis_service.py # sensitivity, VOI
│   └── hybrid_service.py   # continuous nodes, hybrid_query, thresholds
├── viewmodels/             # QObject subclasses; hold state, expose signals
│   ├── graph_viewmodel.py
│   ├── node_inspector_viewmodel.py
│   ├── evidence_viewmodel.py
│   ├── results_viewmodel.py
│   └── main_viewmodel.py
├── views/                  # QWidget / QMainWindow classes only
│   ├── main_window.py
│   ├── graph_canvas/       # QGraphicsScene, node/edge items
│   ├── panels/             # explorer, inspector, results, logs
│   └── dialogs/
├── workers/                # QThread/QRunnable wrappers
│   ├── base_worker.py
│   ├── compile_worker.py
│   ├── query_worker.py
│   └── batch_worker.py
├── commands/               # Undo/redo command stack (QUndoCommand)
│   ├── base.py
│   └── node_commands.py
└── resources/              # icons, qss stylesheets
```

# Layering rules (STRICT)
- `domain/` imports nothing from Qt or `pybncore` internals beyond types.
- `services/` imports `pybncore` and `domain/` only — **never** Qt.
- `viewmodels/` import `services/`, `domain/`, and `PySide6.QtCore` (signals only).
- `views/` import `viewmodels/`, `domain/`, and PySide6 widgets. They **must not** import `services/` or `pybncore`.
- `workers/` import `services/` and `PySide6.QtCore`. They hold no widget references.
- `commands/` import `services/` and `domain/`. They execute/reverse operations via services.

# Data flow (single direction)
User gesture → View → ViewModel method → Service call (sync) or Worker (async) → Service returns DTO → ViewModel signal → View subscribes and re-renders.

# Hard bans
- No `from pybncore ...` inside `views/` or `viewmodels/`.
- No `QWidget` subclass holds a reference to `PyBNCoreWrapper`.
- No global singletons for state — inject the `ModelSession` through constructors.
- No file > ~500 lines. Split by responsibility.
- No business logic inside Qt slots; slots delegate to viewmodel methods.

# Implementation order for any new feature
1. Define/extend a DTO in `domain/results.py` or `domain/*`.
2. Add the service method that returns that DTO.
3. Add a worker if the call can take > 100 ms.
4. Add viewmodel state + signals.
5. Wire the view to the viewmodel.
6. Register an undo command if the action is reversible.

# Review checklist
- Are all imports compliant with the layering rules?
- Is there any `pybncore` symbol in a `views/` file? Reject.
- Is a heavy call running synchronously from a slot? Move it to a worker.
- Is shared state mutated from both UI and worker threads? Route through signals.
